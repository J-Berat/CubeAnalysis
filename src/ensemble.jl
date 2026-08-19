function ensemble_simulation_metrics(directory; box_size=(50.0,50.0,50.0),
        axis_order=("x","y","z"), samples=20_000)
    readfield(name)=read_fits_cube(joinpath(directory,"$name.fits"))
    density=readfield("density"); temperature=readfield("temperature")
    velocity=[readfield(name) for name in ("Vx","Vy","Vz")]
    magnetic=[readfield(name) for name in ("Bx","By","Bz")]
    fields=Dict("density"=>density,"temperature"=>temperature,"Vx"=>velocity[1],"Vy"=>velocity[2],"Vz"=>velocity[3],"Bx"=>magnetic[1],"By"=>magnetic[2],"Bz"=>magnetic[3])
    settings=Dict("density"=>"density","temperature"=>"temperature","velocity"=>["Vx","Vy","Vz"],"magnetic"=>["Bx","By","Bz"],"mean_molecular_weight"=>1.27,"velocity_to_cms"=>1e5,"magnetic_to_gauss"=>1e-3)
    phase=phase_physical_statistics(fields,settings;stride=max(1,length(density)÷samples))
    speed=sqrt.(velocity[1].^2 .+ velocity[2].^2 .+ velocity[3].^2)
    sigma=sqrt(sum(var(vec(component);corrected=false) for component in velocity)/3)
    spectrum=spectrum_details(speed;bins=48,box_size,axis_order)
    physical=inertial_fit_range(nothing,size(speed),box_size,axis_order;injection_modes=4,dissipation_cells=4)
    range=detect_powerlaw_range(spectrum.k,spectrum.shell,physical;min_points=5,min_r2=0.9)
    slope,_,_=spectral_fit(spectrum.k,spectrum.shell,range)
    _,_,_,correlation_length=radial_autocorrelation(density;bins=40,box_size,axis_order)
    phases=default_thermal_phases(); lags,orders,values,counts=phase_velocity_structure(velocity...,temperature,phases;orders=1:5,samples_per_lag=samples)
    zeta=Vector{Float64}[]
    for p in eachindex(phases)
        exponents,_,_=ess_exponents(values[:,p,:],orders,lags,[4,minimum(size(density))÷4];minimum_samples=counts[:,p])
        push!(zeta,exponents)
    end
    global_ms=median(filter(isfinite,vcat([fill(row.mach_sonic,max(row.samples,1)) for row in phase]...)))
    global_ma=median(filter(isfinite,vcat([fill(row.mach_alfven,max(row.samples,1)) for row in phase]...)))
    global_beta=median(filter(isfinite,vcat([fill(row.plasma_beta,max(row.samples,1)) for row in phase]...)))
    return (name=basename(directory),sigma_v=sigma,mach_sonic=global_ms,mach_alfven=global_ma,
        plasma_beta=global_beta,spectral_slope=slope,correlation_length,zeta,
        volume=getfield.(phase,:volume_fraction),mass=getfield.(phase,:mass_fraction))
end

function simulation_phase_ess_metrics(directory;samples=20_000)
    readfield(name)=read_fits_cube(joinpath(directory,"$name.fits"))
    temperature=readfield("temperature")
    velocity=[readfield(name) for name in ("Vx","Vy","Vz")]
    phases=default_thermal_phases()
    lags,orders,values,counts=phase_velocity_structure(velocity...,temperature,phases;
        orders=1:5,samples_per_lag=samples)
    zeta=Vector{Float64}[]
    for phase_index in eachindex(phases)
        exponents,_,_=ess_exponents(values[:,phase_index,:],orders,lags,
            [4,minimum(size(temperature))÷4];minimum_samples=counts[:,phase_index])
        push!(zeta,exponents)
    end
    return (name=basename(directory),zeta)
end

function ensemble_ess_summary(metrics, phase_index, orders=1:5)
    matrix=fill(NaN,length(metrics),length(orders))
    for (simulation_index,metric) in enumerate(metrics)
        values=metric.zeta[phase_index]
        for order_index in eachindex(orders)
            order_index<=length(values) && (matrix[simulation_index,order_index]=values[order_index])
        end
    end
    center=fill(NaN,length(orders)); spread=fill(NaN,length(orders)); samples=zeros(Int,length(orders))
    for order_index in eachindex(orders)
        valid=filter(isfinite,matrix[:,order_index]); samples[order_index]=length(valid)
        isempty(valid) && continue
        center[order_index]=mean(valid)
        spread[order_index]=length(valid)>1 ? std(valid;corrected=true) : 0.0
    end
    return matrix,center,spread,samples
end

function plot_ensemble_phase_ess!(files,metrics,output_dir,formats;overwrite=false)
    phases=default_thermal_phases(); orders=collect(1:5)
    fig=publication_figure(size=(980,680)); axis=latex_axis(fig[1,1],
        xlabel=latexstring("p"),ylabel=latexstring("\\zeta_p/\\zeta_3\\ (\\mathrm{ESS})"),
        xticks=orders)
    for phase_index in eachindex(phases)
        matrix,center,spread,_=ensemble_ess_summary(metrics,phase_index,orders)
        color=series_color(phase_index); marker=series_marker(phase_index)
        for simulation_index in axes(matrix,1)
            valid=findall(index->isfinite(matrix[simulation_index,index]),eachindex(orders))
            length(valid)>=2 || continue
            lines!(axis,orders[valid],matrix[simulation_index,valid];color=(color,0.18),
                linewidth=1.0,linestyle=series_linestyle(phase_index))
        end
        valid=findall(isfinite,center); isempty(valid) && continue
        lines!(axis,orders[valid],center[valid];color,linewidth=2.9,
            label=latex_legend_label(phases[phase_index].name))
        scatter!(axis,orders[valid],center[valid];color=:white,strokecolor=color,
            strokewidth=1.7,markersize=10,marker)
        errorbars!(axis,orders[valid],center[valid],spread[valid];color=(color,0.78),
            linewidth=1.3,whiskerwidth=8)
    end
    order_values=Float64.(orders)
    lines!(axis,order_values,kolmogorov_ess.(order_values);color=PLOT_INK,
        linestyle=:dash,linewidth=2.0,label=latex_legend_label("Kolmogorov"))
    lines!(axis,order_values,she_leveque_ess.(order_values);color=PLOT_MUTED,
        linestyle=:dot,linewidth=2.2,label=latex_legend_label("She-Leveque"))
    lines!(axis,order_values,boldyrev_ess.(order_values);color=PLOT_INK,
        linestyle=:dashdot,linewidth=2.0,label=latex_legend_label("Boldyrev"))
    publication_legend!(axis;position=:lt)
    save_figure!(files,fig,output_dir,"ensemble_phase_ess",formats;overwrite)
    return files
end

function plot_ensemble_comparison!(files,directories,output_dir,formats;overwrite=false,
        box_size=(50.0,50.0,50.0),axis_order=("x","y","z"),samples=20_000)
    stems=("ensemble_physical_scaling","ensemble_phase_fractions","ensemble_phase_ess")
    targets=[joinpath(output_dir,"$stem.$format") for stem in stems for format in formats]
    if !overwrite && all(isfile,targets)
        @info "Ensemble comparison figures already exist; skipping" output_dir
        return files
    end
    physical_missing=overwrite || any(format->!isfile(joinpath(output_dir,
        "ensemble_physical_scaling.$format")),formats)
    fractions_missing=overwrite || any(format->!isfile(joinpath(output_dir,
        "ensemble_phase_fractions.$format")),formats)
    ess_missing=overwrite || any(format->!isfile(joinpath(output_dir,
        "ensemble_phase_ess.$format")),formats)
    if ess_missing && !physical_missing && !fractions_missing
        metrics=[simulation_phase_ess_metrics(directory;samples) for directory in directories]
        plot_ensemble_phase_ess!(files,metrics,output_dir,formats;overwrite)
        return files
    end
    metrics=[ensemble_simulation_metrics(directory;box_size,axis_order,samples) for directory in directories]
    sort!(metrics;by=x->x.sigma_v); x=getfield.(metrics,:sigma_v)
    if physical_missing
        fig=publication_figure(size=(1320,800)); quantities=((:mach_sonic,"\\mathcal{M}_{\\mathrm{s}}"),(:mach_alfven,"\\mathcal{M}_{\\mathrm{A}}"),(:plasma_beta,"\\beta"),(:spectral_slope,"\\alpha_v"),(:correlation_length,"\\ell_{\\mathrm{corr}}\\ [\\mathrm{pc}]"))
        for (panel,(key,label)) in enumerate(quantities)
            ax=latex_axis(fig[cld(panel,3),mod1(panel,3)],xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring(label),xscale=log10)
            values=getfield.(metrics,key); lines!(ax,x,values;color=series_color(panel),linewidth=2.5);scatter!(ax,x,values;color=:white,strokecolor=series_color(panel),strokewidth=1.4,markersize=9)
        end
        save_figure!(files,fig,output_dir,"ensemble_physical_scaling",formats;overwrite)
    end
    phases=default_thermal_phases()
    if fractions_missing
        fig=publication_figure(size=(1280,520))
        for (panel,(key,label)) in enumerate(((:volume,"f_{\\mathrm{V}}"),(:mass,"f_{\\mathrm{M}}")))
            ax=latex_axis(fig[1,panel],xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring(label),xscale=log10)
            for p in eachindex(phases); values=[getfield(row,key)[p] for row in metrics];lines!(ax,x,values;color=series_color(p),linestyle=series_linestyle(p),linewidth=2.5,label=latex_legend_label(phases[p].name));scatter!(ax,x,values;color=series_color(p),marker=series_marker(p),markersize=8);end
            panel==1&&publication_legend!(ax)
        end
        save_figure!(files,fig,output_dir,"ensemble_phase_fractions",formats;overwrite)
    end
    if ess_missing
        plot_ensemble_phase_ess!(files,metrics,output_dir,formats;overwrite)
    end
    return files
end
