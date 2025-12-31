#!/usr/bin/env julia
# ============================================================
# fitter_localparams_v4.jl
#
# BAO-only X(z) reconstruction with low-z-free compression:
#   y = [ DH(z1..zN),  ΔDM(z1→z2 .. z_{N-1}→zN) ] / r_d
#
# HACK/DIAGNOSTIC MODEL:
#   Each X-bin gets its own nuisance parameters:
#     Ωm_j and S_j ≡ (H0 * r_d)_j
#   so the parameter set per bin is (Ωm_j, S_j, X_j), j=1..NB.
#
# This greatly reduces global-parameter-induced correlations among X bins.
# (Residual correlations can remain due to the ΔDM differencing covariance.)
#
# Run:
#   julia fitter_localparams_v4.jl DR1.csv DR2.csv
#
# Outputs (per dataset DR1/DR2):
#   DR*_bin_constraints.csv  (z_lo,z_hi,z_mid, X_mean,X_std, Om_mean,Om_std, S_mean,S_std)
#   DR*_Xcov.csv,  DR*_Xcorr.csv
#   DR*_Omcov.csv, DR*_Omcorr.csv
#   DR*_Scov.csv,  DR*_Scorr.csv
#
# Plots:
#   DESI_DR1_DR2_X_vs_z.pdf         (1×2: DR1 left, DR2 right)
#   DESI_DR1_DR2_X_corr.pdf         (1×2 correlation of X bins)
#   DR*_X_contours.pdf              (corner-style Gaussian ellipses for X bins)
#   DR*_OmegaM_contours.pdf         (corner-style for Ωm bins)
#   DR*_S_contours.pdf              (corner-style for S bins)
#
# External python used only for plotting (matplotlib + numpy required):
#   python3 -m pip install matplotlib numpy
#   (set PYTHON=/path/to/python3 if needed)
#
# Env controls:
#   BAO_INTEGRATOR   = "gauss32" (default) | "gauss64" | "quadgk"
#   NSAMPLES         = post-warmup draws per chain (default 4000)
#   NWARMUP          = warmup steps (default 2000)
#   NCHAINS          = number of chains (default 4)
#   TARGET_ACCEPT    = NUTS target accept (default 0.90)
#   MAX_DEPTH        = NUTS max depth (default 12)
# ============================================================

using Turing, Distributions, LinearAlgebra, Random, MCMCChains, PDMats, StaticArrays
using Statistics, Printf
import ForwardDiff
import QuadGK
using Base.Filesystem: basename

# ------------------------
# Robust CSV-like reader
# Expected columns per row:
#   z, DM, sDM, DH, sDH, r
# ------------------------
function read_bao_csv(path::AbstractString)
    z  = Float64[]; DM  = Float64[]; sDM = Float64[]
    DH = Float64[]; sDH = Float64[]; r   = Float64[]
    for ln0 in eachline(path)
        s = strip(ln0)
        isempty(s) && continue
        startswith(s, "#") && continue
        cols = split(s, r"[,\s;]+"; keepempty=false)
        length(cols) < 6 && continue

        v1 = tryparse(Float64, cols[1]); v1 === nothing && continue
        v2 = tryparse(Float64, cols[2]); v2 === nothing && continue
        v3 = tryparse(Float64, cols[3]); v3 === nothing && continue
        v4 = tryparse(Float64, cols[4]); v4 === nothing && continue
        v5 = tryparse(Float64, cols[5]); v5 === nothing && continue
        v6 = tryparse(Float64, cols[6]); v6 === nothing && continue

        push!(z, v1); push!(DM, v2); push!(sDM, v3)
        push!(DH, v4); push!(sDH, v5); push!(r, v6)
    end
    @assert !isempty(z) "No data rows parsed from $path"

    p = sortperm(z)
    (z=z[p], DM=DM[p], sDM=sDM[p], DH=DH[p], sDH=sDH[p], r=r[p])
end

# ------------------------
# Covariance for raw x = [DM1,DH1, DM2,DH2, ..., DMN,DHN]
# ------------------------
cov2(σDM, σDH, ρ) = @SMatrix [σDM^2  ρ*σDM*σDH;
                               ρ*σDM*σDH  σDH^2]

function build_C_raw(sDM::Vector{Float64}, sDH::Vector{Float64}, r::Vector{Float64})
    N = length(sDM)
    C = zeros(2N, 2N)
    @inbounds for i in 1:N
        ρ = clamp(r[i], -0.9999, 0.9999)
        blk = cov2(sDM[i], sDH[i], ρ)
        C[2i-1:2i, 2i-1:2i] .= blk
    end
    Symmetric(C)
end

# ------------------------
# Non-duplicated transform y = T*x:
#   y = [ DH1..DHN,  ΔDM1..ΔDM_{N−1} ]
# ------------------------
function build_T_transform(N::Int)
    T = zeros(2N-1, 2N)
    @inbounds for i in 1:N
        T[i, 2i] = 1.0
    end
    @inbounds for j in 1:(N-1)
        T[N + j, 2(j+1)-1] =  1.0
        T[N + j, 2j-1]     = -1.0
    end
    T
end

# ------------------------
# Make covariance numerically PD
# ------------------------
function make_PD(Σ::AbstractMatrix; rel=1e-12, max_tries=7)
    S = Symmetric((Σ + Σ')/2)
    try
        return PDMat(Matrix(S))
    catch
    end
    n = size(S,1)
    M = Matrix(S)
    τ = rel * maximum(abs, diag(M))
    τ = max(τ, eps(Float64))
    for _ in 1:max_tries
        Σj = copy(M)
        @inbounds for i in 1:n
            Σj[i,i] += τ
        end
        try
            return PDMat(Σj)
        catch
            τ *= 10
        end
    end
    # last resort: eigenvalue floor
    F = eigen(Symmetric(M))
    λ = clamp.(F.values, τ, Inf)
    Σfix = F.vectors * Diagonal(λ) * F.vectors'
    PDMat(Matrix(Symmetric(Σfix)))
end

# ------------------------
# Background functions
# ------------------------
const c_kms = 299_792.458
A3(z_) = (1 + z_)^3
E2(z_, Ωm_, X_) = Ωm_ * A3(z_) + (1 - Ωm_) * X_
E(z_, Ωm_, X_)  = sqrt(E2(z_, Ωm_, X_))

# ------------------------
# Integrator: Gauss–Legendre (fast) or QuadGK
# ------------------------
function gausslegendre(n::Int)
    n > 0 || error("n must be positive")
    if n == 1
        return ([0.0], [2.0])
    end
    β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    J = SymTridiagonal(zeros(n), β)
    F = eigen(J)
    x = F.values
    w = 2 .* (F.vectors[1, :].^2)
    x, w
end

const INTEGRATOR = lowercase(get(ENV, "BAO_INTEGRATOR", "gauss32"))  # gauss32|gauss64|quadgk
const GL_N = startswith(INTEGRATOR, "gauss") ? something(tryparse(Int, replace(INTEGRATOR, "gauss" => "")), 32) : 0
const GL_x, GL_w = startswith(INTEGRATOR, "gauss") ? gausslegendre(GL_N) : (Float64[], Float64[])

@inline function integral_1_over_E_gauss(Ωm, Xj, a::Float64, b::Float64)
    mid  = (a + b) / 2
    half = (b - a) / 2
    Tacc = typeof(Ωm + Xj)
    acc  = zero(Tacc)
    @inbounds for k in eachindex(GL_x)
        t = mid + half * GL_x[k]
        acc += GL_w[k] / E(t, Ωm, Xj)
    end
    half * acc
end

@inline function integral_1_over_E_quadgk(Ωm, Xj, a::Float64, b::Float64)
    QuadGK.quadgk(t -> 1 / E(t, Ωm, Xj), a, b; rtol=1e-6)[1]
end

const integral_1_over_E = (INTEGRATOR == "quadgk") ? integral_1_over_E_quadgk : integral_1_over_E_gauss

# ------------------------
# Prediction for y with per-bin parameters
# Convention: left-continuous at nodes:
#   DH(z1) uses bin1; for i>=2, DH(zi) uses bin (i-1)
# Each ΔDM_j uses bin j
# ------------------------
function mu_from_params_local(Ωm::AbstractVector, S::AbstractVector, X::AbstractVector, z::Vector{Float64})
    N  = length(z)
    NB = N - 1
    @assert length(Ωm)==NB && length(S)==NB && length(X)==NB

    first_entry = (c_kms / S[1]) * (1 / E(z[1], Ωm[1], X[1]))
    Tμ = typeof(first_entry)
    μ  = Vector{Tμ}(undef, 2N-1)

    # DH nodes
    μ[1] = first_entry
    @inbounds for i in 2:N
        b = i - 1
        μ[i] = (c_kms / S[b]) * (1 / E(z[i], Ωm[b], X[b]))
    end

    # ΔDM bins
    @inbounds for j in 1:NB
        pref = c_kms / S[j]
        I    = integral_1_over_E(Ωm[j], X[j], z[j], z[j+1])
        μ[N + j] = pref * I
    end
    μ
end

# ------------------------
# Turing model (per-bin Ωm and S)
# ------------------------
@model function bao_nodup_local(y, Σ::PDMat, z::Vector{Float64})
    N  = length(z)
    NB = N - 1

#    Ωm ~ filldist(Uniform(0.2, 0.4), NB)
#    S  ~ filldist(Uniform(9.2e3, 1.06e4), NB)
    Ωm ~ filldist(Uniform(0.28, 0.35), NB)
    S  ~ filldist(Uniform(9.423e3, 1.0393e4), NB)
    X  ~ filldist(Uniform(-2.0, 4.0), NB)

    @inbounds for j in 1:NB
        if (E2(z[j],   Ωm[j], X[j]) <= 0.0) || (E2(z[j+1], Ωm[j], X[j]) <= 0.0)
            Turing.@addlogprob!(-Inf)
            return
        end
    end

    μ = mu_from_params_local(Ωm, S, X, z)
    y ~ MvNormal(μ, Σ)
    return nothing
end

# Prefer ForwardDiff AD
if isdefined(Turing, :setadbackend!)
    Turing.setadbackend!(Turing.ForwardDiffAD())
elseif isdefined(Turing, :setadbackend)
    Turing.setadbackend(:forwarddiff)
end
if isdefined(Turing, :setrdcache)
    Turing.setrdcache(true)
end

# ------------------------
# Helpers to extract parameter matrices from chains
# ------------------------
function extract_matrix_from_chains(ch::Chains, p_syms::Vector{Symbol}, sym_prefix::String, NB::Int)
    vals = ch.value  # (nsamp, nparams, nchains)
    ns = size(vals, 1)
    nc = size(vals, 3)
    M  = Matrix{Float64}(undef, ns*nc, NB)
    @inbounds for j in 1:NB
        sym = Symbol("$(sym_prefix)[$j]")
        idx = findfirst(==(sym), p_syms)
        @assert idx !== nothing "Parameter $sym not found in chains"
        v = @view vals[:, idx, :]
        M[:, j] = vec(v)
    end
    M
end

# ------------------------
# Fit one dataset and return summaries + cov/corr
# ------------------------
function fit_one(csv_path::AbstractString; seed::Int=20250909)
    d = read_bao_csv(csv_path)
    z, DM, sDM, DH, sDH, r = d.z, d.DM, d.sDM, d.DH, d.sDH, d.r
    N  = length(z)
    NB = N - 1
    zmid = 0.5 .* (z[1:NB] .+ z[2:N])

    x = zeros(2N)
    @inbounds for i in 1:N
        x[2i-1] = DM[i]
        x[2i]   = DH[i]
    end

    C  = build_C_raw(sDM, sDH, r)
    T  = build_T_transform(N)
    y  = T * x
    Σy = Symmetric(T * Matrix(C) * transpose(T))
    ΣyPD = make_PD(Σy)

    nsamp   = parse(Int, get(ENV, "NSAMPLES", "4000"))
    nwarm   = parse(Int, get(ENV, "NWARMUP", "2000"))
    nchains = parse(Int, get(ENV, "NCHAINS", "4"))
    targacc = parse(Float64, get(ENV, "TARGET_ACCEPT", "0.90"))
    maxdep  = parse(Int, get(ENV, "MAX_DEPTH", "12"))

    Random.seed!(seed)
    println("\n--- Fitting: $(basename(csv_path)) ---")
    println("Integrator: ", INTEGRATOR == "quadgk" ? "QuadGK rtol=1e-6" : "Gauss–Legendre (N=$(GL_N))")
    println("NUTS: nsamples=$nsamp, nwarmup=$nwarm, nchains=$nchains, target_accept=$targacc, max_depth=$maxdep")
    println("Model: per-bin Ωm_j and S_j (H0*rd)")

    ch = sample(bao_nodup_local(y, ΣyPD, z),
                NUTS(targacc; max_depth=maxdep),
                MCMCThreads(),
                nsamp, nchains;
                num_warmup=nwarm, discard_adapt=true, progress=true)

    p_syms = Symbol.(names(ch, :parameters))

    Xsamp  = extract_matrix_from_chains(ch, p_syms, "X", NB)
    Omsamp = extract_matrix_from_chains(ch, p_syms, "Ωm", NB)
    Ssamp  = extract_matrix_from_chains(ch, p_syms, "S", NB)

    Xmean = vec(mean(Xsamp, dims=1)); Xstd = vec(std(Xsamp, dims=1))
    Ommean = vec(mean(Omsamp, dims=1)); Omstd = vec(std(Omsamp, dims=1))
    Smean = vec(mean(Ssamp, dims=1));  Sstd  = vec(std(Ssamp, dims=1))

    Xcov  = cov(Xsamp; dims=1) |> Matrix
    Xcorr = cor(Xsamp) |> Matrix
    Omcov  = cov(Omsamp; dims=1) |> Matrix
    Omcorr = cor(Omsamp) |> Matrix
    Scov  = cov(Ssamp; dims=1) |> Matrix
    Scorr = cor(Ssamp) |> Matrix

    println("\nPosterior mean ± std per bin:")
    @inbounds for j in 1:NB
        @printf("  bin %d: X=%.6f ± %.6f,  Ωm=%.6f ± %.6f,  S=%.3f ± %.3f (km/s)\n",
                j, Xmean[j], Xstd[j], Ommean[j], Omstd[j], Smean[j], Sstd[j])
    end
    if NB >= 2
        off = [abs(Xcorr[i,j]) for i in 1:NB for j in 1:NB if i!=j]
        @printf("Mean |corr_X offdiag| = %.4f (median=%.4f)\n", mean(off), median(off))
    end

    # free heavy objects
    ch = nothing
    GC.gc()

    return (z=z, zmid=zmid, NB=NB,
            Xmean=Xmean, Xstd=Xstd, Xcov=Xcov, Xcorr=Xcorr,
            Ommean=Ommean, Omstd=Omstd, Omcov=Omcov, Omcorr=Omcorr,
            Smean=Smean, Sstd=Sstd, Scov=Scov, Scorr=Scorr)
end

# ------------------------
# CSV writers
# ------------------------
function write_bin_constraints_csv(path::AbstractString, z::Vector{Float64},
                                   Xmean::Vector{Float64}, Xstd::Vector{Float64},
                                   Ommean::Vector{Float64}, Omstd::Vector{Float64},
                                   Smean::Vector{Float64},  Sstd::Vector{Float64})
    N  = length(z)
    NB = N - 1
    @assert length(Xmean)==NB && length(Xstd)==NB
    @assert length(Ommean)==NB && length(Omstd)==NB
    @assert length(Smean)==NB && length(Sstd)==NB
    open(path, "w") do io
        println(io, "bin,z_lo,z_hi,z_mid,X_mean,X_std,Om_mean,Om_std,S_mean,S_std")
        @inbounds for j in 1:NB
            zlo = z[j]; zhi = z[j+1]; zm = 0.5*(zlo+zhi)
            @printf(io, "%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n",
                    j, zlo, zhi, zm, Xmean[j], Xstd[j], Ommean[j], Omstd[j], Smean[j], Sstd[j])
        end
    end
end

function write_matrix_csv(path::AbstractString, M::AbstractMatrix{<:Real})
    open(path, "w") do io
        for i in 1:size(M,1)
            println(io, join([@sprintf("%.16g", M[i,j]) for j in 1:size(M,2)], ","))
        end
    end
end

dataset_label(path::AbstractString) = occursin("DR1", uppercase(path)) ? "DESI DR1" :
                                     occursin("DR2", uppercase(path)) ? "DESI DR2" : basename(path)

# ------------------------
# Robust python plotting runner
# ------------------------
function run_python_plot(script_text::AbstractString, out_pdf::AbstractString; args=String[])
    python = get(ENV, "PYTHON", "python3")
    tmpdir = mktempdir()
    pyscr  = joinpath(tmpdir, "plot.py")
    tmp_pdf = joinpath(tmpdir, basename(out_pdf))
    write(pyscr, script_text)

    argv = String[python, pyscr]
    append!(argv, String.(args))
    push!(argv, tmp_pdf)
    run(Cmd(argv))

    cp(tmp_pdf, out_pdf; force=true)
    println("Saved: ", out_pdf)
end

# ------------------------
# Two-panel X(z) plot (uses DR*_bin_constraints.csv)
# ------------------------
function plot_Xz_two_panel(csv1::AbstractString, lab1::AbstractString,
                           csv2::AbstractString, lab2::AbstractString,
                           out_pdf::AbstractString)
    script = raw"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Times for text, CM for math
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman","Times","Nimbus Roman","DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

csv1, lab1, csv2, lab2, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def read_bins(path):
    zlo=[]; zhi=[]; zm=[]; xm=[]; xs=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            zlo.append(float(row["z_lo"]))
            zhi.append(float(row["z_hi"]))
            zm.append(float(row["z_mid"]))
            xm.append(float(row["X_mean"]))
            xs.append(float(row["X_std"]))
    zlo=np.array(zlo); zhi=np.array(zhi); zm=np.array(zm)
    xm=np.array(xm); xs=np.array(xs)
    xerrm = zm - zlo
    xerrp = zhi - zm
    return zm, xm, xs, xerrm, xerrp

z1, x1, s1, xerrm1, xerrp1 = read_bins(csv1)
z2, x2, s2, xerrm2, xerrp2 = read_bins(csv2)

# Wide figure for full-width A4 insertion
fig, axs = plt.subplots(1,2, figsize=(11.8,4.9), dpi=280, sharey=True)

# Leave room for a centered shared xlabel inside the canvas
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.28, top=0.93, wspace=0.10)

color = "#1f77b4"

def style(ax):
    ax.tick_params(axis="both", which="both",
                   direction="in",
                   top=True, right=True,
                   labeltop=False, labelright=False,
                   labelsize=18,
                   width=1.2, length=6)
    for sp in ("top","right","bottom","left"):
        ax.spines[sp].set_linewidth(1.2)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axhline(1.0, linestyle="--", linewidth=2.2, color="#d95f02", alpha=0.9)

# left panel
axs[0].errorbar(z1, x1,
    xerr=[xerrm1,xerrp1],
    yerr=s1,
    fmt="o",
    markersize=7.2,
    markerfacecolor=color,
    markeredgecolor="black",
    markeredgewidth=0.8,
    ecolor=color,
    elinewidth=2.2,
    capsize=4.2,
    capthick=2.2,
    alpha=0.95)
style(axs[0])
axs[0].set_title(lab1, fontsize=20, pad=8)

# right panel
axs[1].errorbar(z2, x2,
    xerr=[xerrm2,xerrp2],
    yerr=s2,
    fmt="o",
    markersize=7.2,
    markerfacecolor=color,
    markeredgecolor="black",
    markeredgewidth=0.8,
    ecolor=color,
    elinewidth=2.2,
    capsize=4.2,
    capthick=2.2,
    alpha=0.95)
style(axs[1])
axs[1].set_title(lab2, fontsize=20, pad=8)

# y-label on left only (avoid crowding in the middle)
axs[0].set_ylabel(r"$X(z)$", fontsize=28, labelpad=12)

# --- Center the shared xlabel w.r.t. the union of the two axes ---
fig.canvas.draw()  # ensure positions are updated

# Keep edge tick labels inside each panel to avoid clipping at the figure boundary
for ax in axs:
    xt = ax.get_xticklabels()
    if len(xt) >= 2:
        xt[0].set_ha("left")
        xt[-1].set_ha("right")
    for t in xt:
        t.set_clip_on(False)

pos0 = axs[0].get_position()
pos1 = axs[1].get_position()
xcenter = 0.5 * (pos0.x0 + pos1.x1)
fig.text(xcenter, 0.12, r"redshift $z$", ha="center", va="center", fontsize=28)

# Save WITHOUT bbox_inches='tight' so the centering isn't distorted by cropping
fig.savefig(out_pdf)
plt.close(fig)
"""
    run_python_plot(script, out_pdf; args=[csv1, lab1, csv2, lab2])
end

# ------------------------
# Two-panel correlation heatmap for X bins
# ------------------------
function plot_corr_two_panel(corr1_path::AbstractString, lab1::AbstractString,
                             corr2_path::AbstractString, lab2::AbstractString,
                             out_pdf::AbstractString)
    script = raw"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman","Times","Nimbus Roman","DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

p1, lab1, p2, lab2, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
C1 = np.loadtxt(p1, delimiter=",")
C2 = np.loadtxt(p2, delimiter=",")

fig, axs = plt.subplots(1,2, figsize=(11.5,4.6), dpi=260)
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90, wspace=0.16)

def one(ax, C, lab):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm", origin="lower", interpolation="nearest")
    nb = C.shape[0]
    ax.set_xticks(range(nb))
    ax.set_yticks(range(nb))
    ax.set_xticklabels([f"{i+1}" for i in range(nb)], fontsize=15)
    ax.set_yticklabels([f"{i+1}" for i in range(nb)], fontsize=15)
    ax.set_xlabel(r"bin index", fontsize=18)
    ax.set_ylabel(r"bin index", fontsize=18)
    ax.set_title(lab, fontsize=20, pad=8)
    ax.tick_params(direction="in", top=True, right=True, labeltop=False, labelright=False, width=1.0, length=4)
    for sp in ("top","right","bottom","left"):
        ax.spines[sp].set_linewidth(1.0)
    return im

im1 = one(axs[0], C1, lab1)
im2 = one(axs[1], C2, lab2)

cax = fig.add_axes([0.975, 0.20, 0.015, 0.62])
cb = fig.colorbar(im2, cax=cax)
cb.set_label("corr", fontsize=16)
cb.ax.tick_params(labelsize=13)

fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
"""
    run_python_plot(script, out_pdf; args=[corr1_path, lab1, corr2_path, lab2])
end

# ------------------------
# Corner-style Gaussian ellipse plot for a vector parameter
# Inputs:
#   mean.csv: one line of NB numbers (or NB lines okay)
#   cov.csv : NB×NB
# ------------------------
function plot_corner_gaussian(mean_csv::AbstractString, cov_csv::AbstractString,
                              labels::Vector{String}, title::String, out_pdf::AbstractString)
    @assert !isempty(labels)
    lab_join = join(labels, "|")  # pass as single arg
    script = raw"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman","Times","Nimbus Roman","DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

mean_path, cov_path, labstr, titlestr, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
labels = labstr.split("|")

m = np.loadtxt(mean_path, delimiter=",")
m = np.atleast_1d(m).reshape(-1)
C = np.loadtxt(cov_path, delimiter=",")
nb = C.shape[0]
assert len(m) == nb
assert len(labels) == nb

# 2D chi2 for 68% and 95% in 2 dof
chi2_68 = 2.30
chi2_95 = 6.17

def ellipse_from_cov(mu, cov2, level, **kwargs):
    # cov2 2x2
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # angle in degrees
    ang = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width  = 2.0 * np.sqrt(vals[0] * level)
    height = 2.0 * np.sqrt(vals[1] * level)
    return Ellipse(xy=mu, width=width, height=height, angle=ang, fill=False, **kwargs)

# Figure
fig, axs = plt.subplots(nb, nb, figsize=(2.05*nb, 2.05*nb), dpi=220)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.92, wspace=0.05, hspace=0.05)
fig.suptitle(titlestr, fontsize=16, y=0.995)

for i in range(nb):
    for j in range(nb):
        ax = axs[i,j]
        if i < j:
            ax.axis("off")
            continue

        ax.tick_params(direction="in", top=True, right=True, labelsize=9, width=0.9, length=3)
        for sp in ("top","right","bottom","left"):
            ax.spines[sp].set_linewidth(0.9)

        if i == j:
            mu = m[i]
            sig = np.sqrt(C[i,i])
            xs = np.linspace(mu - 4*sig, mu + 4*sig, 400)
            ys = np.exp(-0.5*((xs-mu)/sig)**2) / (sig*np.sqrt(2*np.pi))
            ax.plot(xs, ys, lw=1.6, color="#1f77b4")
            ax.axvline(mu, lw=1.2, color="k", alpha=0.7)
            ax.set_yticks([])
        else:
            mu2 = np.array([m[j], m[i]])
            cov2 = np.array([[C[j,j], C[j,i]],
                             [C[i,j], C[i,i]]])
            e1 = ellipse_from_cov(mu2, cov2, chi2_68, lw=1.6, edgecolor="#1f77b4")
            e2 = ellipse_from_cov(mu2, cov2, chi2_95, lw=1.3, edgecolor="#1f77b4", linestyle="--")
            ax.add_patch(e2)
            ax.add_patch(e1)

        # labels only on left column and bottom row
        if i == nb-1:
            ax.set_xlabel(labels[j], fontsize=11)
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(labels[i], fontsize=11)
        elif j != 0:
            ax.set_yticklabels([])

fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
"""
    # write temporary mean/cov to ensure correct delimiter
    tmpdir = mktempdir()
    mtmp = joinpath(tmpdir, "mean.csv")
    ctmp = joinpath(tmpdir, "cov.csv")
    # copy to tmp with comma delimiter guaranteed
    cp(mean_csv, mtmp; force=true)
    cp(cov_csv,  ctmp; force=true)
    run_python_plot(script, out_pdf; args=[mtmp, ctmp, lab_join, title])
end

# ------------------------
# Main
# ------------------------
if length(ARGS) < 2
    println("Usage: julia fitter_localparams_v4.jl DR1.csv DR2.csv")
    exit(1)
end

csv1, csv2 = ARGS[1], ARGS[2]
lab1 = dataset_label(csv1)
lab2 = dataset_label(csv2)

# ---- Fit DR1 ----
r1 = fit_one(csv1; seed=20250909)
write_bin_constraints_csv("DR1_bin_constraints.csv", r1.z, r1.Xmean, r1.Xstd, r1.Ommean, r1.Omstd, r1.Smean, r1.Sstd)
write_matrix_csv("DR1_Xcov.csv",  r1.Xcov)
write_matrix_csv("DR1_Xcorr.csv", r1.Xcorr)
write_matrix_csv("DR1_Omcov.csv", r1.Omcov)
write_matrix_csv("DR1_Omcorr.csv", r1.Omcorr)
write_matrix_csv("DR1_Scov.csv",  r1.Scov)
write_matrix_csv("DR1_Scorr.csv", r1.Scorr)

# mean vectors for corner plots
open("DR1_Xmean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r1.Xmean], ",")); end
open("DR1_Ommean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r1.Ommean], ",")); end
open("DR1_Smean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r1.Smean], ",")); end

GC.gc()

# ---- Fit DR2 ----
r2 = fit_one(csv2; seed=20250910)
write_bin_constraints_csv("DR2_bin_constraints.csv", r2.z, r2.Xmean, r2.Xstd, r2.Ommean, r2.Omstd, r2.Smean, r2.Sstd)
write_matrix_csv("DR2_Xcov.csv",  r2.Xcov)
write_matrix_csv("DR2_Xcorr.csv", r2.Xcorr)
write_matrix_csv("DR2_Omcov.csv", r2.Omcov)
write_matrix_csv("DR2_Omcorr.csv", r2.Omcorr)
write_matrix_csv("DR2_Scov.csv",  r2.Scov)
write_matrix_csv("DR2_Scorr.csv", r2.Scorr)

open("DR2_Xmean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r2.Xmean], ",")); end
open("DR2_Ommean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r2.Ommean], ",")); end
open("DR2_Smean.csv","w") do io; println(io, join([@sprintf("%.16g", v) for v in r2.Smean], ",")); end

GC.gc()

# ---- Plots comparing DR1 vs DR2 ----
plot_Xz_two_panel("DR1_bin_constraints.csv", lab1, "DR2_bin_constraints.csv", lab2, "DESI_DR1_DR2_X_vs_z.pdf")
plot_corr_two_panel("DR1_Xcorr.csv", lab1, "DR2_Xcorr.csv", lab2, "DESI_DR1_DR2_X_corr.pdf")

# ---- Corner/contour plots ----
# X
labelsX1 = [ "\$X_{$i}\$" for i in 1:r1.NB ]
labelsX2 = [ "\$X_{$i}\$" for i in 1:r2.NB ]
plot_corner_gaussian("DR1_Xmean.csv", "DR1_Xcov.csv", labelsX1, "DESI DR1: X bins (Gaussian ellipses)", "DR1_X_contours.pdf")
plot_corner_gaussian("DR2_Xmean.csv", "DR2_Xcov.csv", labelsX2, "DESI DR2: X bins (Gaussian ellipses)", "DR2_X_contours.pdf")

# Ωm
labelsOm1 = [ "\$\\Omega_{m,$i}\$" for i in 1:r1.NB ]
labelsOm2 = [ "\$\\Omega_{m,$i}\$" for i in 1:r2.NB ]
plot_corner_gaussian("DR1_Ommean.csv", "DR1_Omcov.csv", labelsOm1, "DESI DR1: bin-by-bin \$\\Omega_m\$ (Gaussian ellipses)", "DR1_OmegaM_contours.pdf")
plot_corner_gaussian("DR2_Ommean.csv", "DR2_Omcov.csv", labelsOm2, "DESI DR2: bin-by-bin \$\\Omega_m\$ (Gaussian ellipses)", "DR2_OmegaM_contours.pdf")

# S
labelsS1 = [ "\$S_{$i}\$" for i in 1:r1.NB ]
labelsS2 = [ "\$S_{$i}\$" for i in 1:r2.NB ]
plot_corner_gaussian("DR1_Smean.csv", "DR1_Scov.csv", labelsS1, "DESI DR1: bin-by-bin \$S\\equiv H_0 r_d\$ (Gaussian ellipses)", "DR1_S_contours.pdf")
plot_corner_gaussian("DR2_Smean.csv", "DR2_Scov.csv", labelsS2, "DESI DR2: bin-by-bin \$S\\equiv H_0 r_d\$ (Gaussian ellipses)", "DR2_S_contours.pdf")

println("\nAll done.")

