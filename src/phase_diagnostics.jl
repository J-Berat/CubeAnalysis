function phase_index(value, phases)
    return findfirst(phase -> phase.minimum <= value < phase.maximum, phases)
end

phase_tick_formatter(names) = values -> [latex_legend_label(
    1 <= round(Int,value) <= length(names) ? names[round(Int,value)] : "") for value in values]

function phase_physical_statistics(fields, settings; stride=1, blocks_per_axis=4,
        bootstrap_replicates=200, seed=1234)
    phases = configured_thermal_phases(settings)
    density = fields[string(get(settings, "density", "density"))]
    temperature = fields[string(get(settings, "temperature", "temperature"))]
    velocity = [fields[name] for name in string.(get(settings, "velocity", ["Vx", "Vy", "Vz"]))]
    magnetic = [fields[name] for name in string.(get(settings, "magnetic", ["Bx", "By", "Bz"]))]
    mu=Float64(get(settings,"mean_molecular_weight",1.27)); gamma=Float64(get(settings,"gamma",5/3))
    vunit=Float64(get(settings,"velocity_to_cms",1e5)); bunit=Float64(get(settings,"magnetic_to_gauss",1e-3))
    kB=1.380649e-16; mH=1.6735575e-24
    counts=zeros(Int,length(phases)); masses=zeros(Float64,length(phases))
    block_count=zeros(Int,blocks_per_axis^3,length(phases)); block_mass=zeros(Float64,blocks_per_axis^3,length(phases))
    sigma_sums=zeros(Float64,length(phases)); ms=[Float64[] for _ in phases]
    ma=[Float64[] for _ in phases]; beta=[Float64[] for _ in phases]; bmag=[Float64[] for _ in phases]
    total_count=0; total_mass=0.0
    for index in 1:stride:length(density)
        n=Float64(density[index]); T=Float64(temperature[index])
        p=phase_index(T,phases)
        if isnothing(p) || !(isfinite(n)&&isfinite(T)&&n>0&&T>0)
            continue
        end
        vv=sum(abs2,Float64(component[index]) for component in velocity)
        bb=sum(abs2,Float64(component[index]) for component in magnetic)
        isfinite(vv+bb) || continue
        counts[p]+=1; masses[p]+=n; total_count+=1; total_mass+=n
        coordinate=CartesianIndices(density)[index]
        block_coordinate=ntuple(d->min(blocks_per_axis,cld(coordinate[d]*blocks_per_axis,size(density,d))),3)
        block=block_coordinate[1]+blocks_per_axis*(block_coordinate[2]-1)+blocks_per_axis^2*(block_coordinate[3]-1)
        block_count[block,p]+=1; block_mass[block,p]+=n
        sigma_sums[p]+=vv/3; B=sqrt(bb); push!(bmag[p],B)
        sound=sqrt(gamma*kB*T/(mu*mH)); rho=mu*mH*n
        push!(ms[p],vunit*sqrt(vv)/sound)
        if B>0
            Bg=bunit*B; alfven=Bg/sqrt(4π*rho)
            push!(ma[p],vunit*sqrt(vv)/alfven); push!(beta[p],8π*n*kB*T/Bg^2)
        end
    end
    rng=MersenneTwister(seed); volume_boot=[Float64[] for _ in phases]; mass_boot=[Float64[] for _ in phases]
    for _ in 1:bootstrap_replicates
        selected=rand(rng,1:size(block_count,1),size(block_count,1)); totals=vec(sum(block_count[selected,:];dims=1)); weights=vec(sum(block_mass[selected,:];dims=1))
        for p in eachindex(phases);push!(volume_boot[p],totals[p]/max(sum(totals),1));push!(mass_boot[p],weights[p]/max(sum(weights),eps()));end
    end
    return [(name=phases[p].name, volume_fraction=counts[p]/max(total_count,1),
        mass_fraction=masses[p]/max(total_mass,eps()),
        volume_std=std(volume_boot[p];corrected=false),mass_std=std(mass_boot[p];corrected=false),
        sigma_v=sqrt(sigma_sums[p]/max(counts[p],1)),
        mach_sonic=isempty(ms[p]) ? NaN : median(ms[p]),
        mach_alfven=isempty(ma[p]) ? NaN : median(ma[p]),
        plasma_beta=isempty(beta[p]) ? NaN : median(beta[p]),
        magnetic=isempty(bmag[p]) ? NaN : median(bmag[p]), samples=counts[p]) for p in eachindex(phases)]
end

function phase_increment_statistics(vx,vy,vz,temperature,phases; lags=nothing,
        samples_per_lag=100_000,seed=1234,periodic=true)
    shape=size(vx); selected=isnothing(lags) ? unique!(round.(Int,10 .^ range(0,log10(max(1,minimum(shape)÷2));length=14))) : Int.(lags)
    sums=zeros(Float64,length(selected),length(phases),3); counts=zeros(Int,length(selected),length(phases))
    pdf_samples=[Float64[] for _ in selected, _ in phases]; rng=MersenneTwister(seed)
    for (li,lag) in enumerate(selected), _ in 1:samples_per_lag
        axis=rand(rng,1:3); direction=rand(rng,Bool) ? lag : -lag
        source=ntuple(d->rand(rng,axes(vx,d)),3); target=collect(source); shifted=source[axis]+direction
        if periodic; target[axis]=mod1(shifted,shape[axis]); elseif 1<=shifted<=shape[axis]; target[axis]=shifted; else; continue; end
        T1=Float64(temperature[source...]); T2=Float64(temperature[target...])
        p=phase_index(T1,phases); isnothing(p) && continue
        p==phase_index(T2,phases) || continue
        components=(vx,vy,vz); du=Float64(components[axis][target...]-components[axis][source...])
        isfinite(du) || continue
        sums[li,p,1]+=du^2; sums[li,p,2]+=du^3; sums[li,p,3]+=du^4
        counts[li,p]+=1
        length(pdf_samples[li,p])<5000 && push!(pdf_samples[li,p],du)
    end
    skew=fill(NaN,length(selected),length(phases)); flat=copy(skew)
    for li in eachindex(selected),p in eachindex(phases)
        n=counts[li,p]; n>2 || continue; m2=sums[li,p,1]/n
        m2>0 || continue; skew[li,p]=(sums[li,p,2]/n)/m2^1.5
        flat[li,p]=(sums[li,p,3]/n)/m2^2
    end
    return selected,skew,flat,counts,pdf_samples
end

function plot_phase_diagnostics!(files,fields,metadata,cfg,output_dir,formats,overwrite)
    settings=get(cfg,"diagnostics",Dict()); Bool(get(settings,"phase_diagnostics",true)) || return files
    stats=phase_physical_statistics(fields,settings;stride=Int(get(cfg["run"],"sample_stride",1)),
        blocks_per_axis=Int(get(settings,"bootstrap_blocks_per_axis",4)),bootstrap_replicates=Int(get(settings,"bootstrap_replicates",200)),seed=Int(get(settings,"seed",1234)))
    names=getfield.(stats,:name); positions=collect(1:length(stats)); phase_ticks=phase_tick_formatter(names)
    fig=publication_figure(size=(920,520)); ax=latex_axis(fig[1,1],xlabel="thermal phase",ylabel="fraction",xticks=positions,xtickformat=phase_ticks)
    barplot!(ax,positions .- 0.18,getfield.(stats,:volume_fraction);width=0.34,color=PLOT_BLUE,label=latexstring("f_{\\mathrm{V}}"))
    barplot!(ax,positions .+ 0.18,getfield.(stats,:mass_fraction);width=0.34,color=PLOT_ORANGE,label=latexstring("f_{\\mathrm{M}}"))
    errorbars!(ax,positions .- 0.18,getfield.(stats,:volume_fraction),getfield.(stats,:volume_std);color=PLOT_INK,whiskerwidth=8)
    errorbars!(ax,positions .+ 0.18,getfield.(stats,:mass_fraction),getfield.(stats,:mass_std);color=PLOT_INK,whiskerwidth=8)
    ylims!(ax,0,nothing); publication_legend!(ax)
    save_figure!(files,fig,output_dir,"phase_fractions",formats;overwrite)

    quantities=((:sigma_v,"\\sigma_v"),(:mach_sonic,"\\mathcal{M}_{\\mathrm{s}}"),
        (:mach_alfven,"\\mathcal{M}_{\\mathrm{A}}"),(:plasma_beta,"\\beta"),(:magnetic,"|B|"))
    fig=publication_figure(size=(1320,760))
    for (panel,(key,label)) in enumerate(quantities)
        ax=latex_axis(fig[cld(panel,3),mod1(panel,3)],xlabel="thermal phase",ylabel=latexstring(label),yscale=log10,xticks=positions,xtickformat=phase_ticks)
        values=getfield.(stats,key); valid=isfinite.(values).&(values.>0)
        scatter!(ax,positions[valid],values[valid];markersize=14,color=series_color(panel),marker=series_marker(panel))
    end
    save_figure!(files,fig,output_dir,"phase_physical_diagnostics",formats;overwrite)

    phases=configured_thermal_phases(settings); names=string.(get(settings,"phase_ess_velocity",["Vx","Vy","Vz"])); temp=string(get(settings,"phase_ess_temperature","temperature"))
    lags,skew,flat,_,pdfs=phase_increment_statistics((fields[n] for n in names)...,fields[temp],phases;
        samples_per_lag=Int(get(settings,"phase_increment_samples",50_000)),seed=Int(get(settings,"seed",1234)),periodic=Bool(get(get(cfg,"grid",Dict()),"periodic",true)))
    spacing=mean(grid_spacings(fields[names[1]],get(get(cfg,"grid",Dict()),"box_size",1.0);axis_order=string.(get(get(cfg,"grid",Dict()),"axis_order",["x","y","z"]))))
    separation=lags.*spacing; fig=publication_figure(size=(920,470))
    for (panel,(values,label)) in enumerate(((skew,"\\mathcal{S}"),(flat,"\\mathcal{F}")))
        ax=latex_axis(fig[1,panel],xlabel=latexstring("\\ell\\ [\\mathrm{pc}]"),ylabel=latexstring(label),xscale=log10)
        for p in eachindex(phases); lines!(ax,separation,values[:,p];color=series_color(p),linestyle=series_linestyle(p),linewidth=2.5,label=latex_legend_label(phases[p].name)); end
        panel==1 && publication_legend!(ax)
    end
    save_figure!(files,fig,output_dir,"phase_intermittency",formats;overwrite)

    chosen=unique(clamp.(round.(Int,range(1,length(lags);length=min(3,length(lags)))),1,length(lags)))
    fig=publication_figure(size=(440*length(chosen),480))
    for (panel,li) in enumerate(chosen)
        ax=latex_axis(fig[1,panel],xlabel=latexstring("\\delta v_{\\mathrm{L}}/\\sigma_{\\delta v}"),ylabel=latexstring("p(\\delta v_{\\mathrm{L}}/\\sigma_{\\delta v})"),yscale=log10)
        for p in eachindex(phases)
            values=pdfs[li,p]; length(values)>10 || continue; standardized=(values.-mean(values))./std(values)
            edges=collect(range(-6,6;length=61)); counts=zeros(Float64,60)
            for value in standardized; -6<=value<=6 || continue; counts[clamp(searchsortedlast(edges,value),1,60)]+=1; end
            centers=0.5.*(edges[1:end-1].+edges[2:end]); counts./=max(sum(counts)*(edges[2]-edges[1]),eps())
            valid=counts.>0
            lines!(ax,centers[valid],counts[valid];color=series_color(p),linestyle=series_linestyle(p),linewidth=2.4,label=latex_legend_label(phases[p].name))
        end
        panel==1 && publication_legend!(ax)
    end
    save_figure!(files,fig,output_dir,"phase_increment_pdfs",formats;overwrite)
    return files
end
