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
    zeta2=Float64[]
    for p in eachindex(phases)
        exponents,_,_=ess_exponents(values[:,p,:],orders,lags,[4,minimum(size(density))÷4];minimum_samples=counts[:,p])
        push!(zeta2,exponents[2])
    end
    global_ms=median(filter(isfinite,vcat([fill(row.mach_sonic,max(row.samples,1)) for row in phase]...)))
    global_ma=median(filter(isfinite,vcat([fill(row.mach_alfven,max(row.samples,1)) for row in phase]...)))
    global_beta=median(filter(isfinite,vcat([fill(row.plasma_beta,max(row.samples,1)) for row in phase]...)))
    return (name=basename(directory),sigma_v=sigma,mach_sonic=global_ms,mach_alfven=global_ma,
        plasma_beta=global_beta,spectral_slope=slope,correlation_length,zeta2,
        volume=getfield.(phase,:volume_fraction),mass=getfield.(phase,:mass_fraction))
end

function plot_ensemble_comparison!(files,directories,output_dir,formats;overwrite=false,
        box_size=(50.0,50.0,50.0),axis_order=("x","y","z"),samples=20_000)
    targets=[joinpath(output_dir,"$stem.$format") for stem in
        ("ensemble_physical_scaling","ensemble_phase_fractions","ensemble_phase_zeta2") for format in formats]
    if !overwrite && all(isfile,targets)
        @info "Ensemble comparison figures already exist; skipping" output_dir
        return files
    end
    metrics=[ensemble_simulation_metrics(directory;box_size,axis_order,samples) for directory in directories]
    sort!(metrics;by=x->x.sigma_v); x=getfield.(metrics,:sigma_v)
    fig=publication_figure(size=(1320,800)); quantities=((:mach_sonic,"\\mathcal{M}_{\\mathrm{s}}"),(:mach_alfven,"\\mathcal{M}_{\\mathrm{A}}"),(:plasma_beta,"\\beta"),(:spectral_slope,"\\alpha_v"),(:correlation_length,"\\ell_{\\mathrm{corr}}\\ [\\mathrm{pc}]"))
    for (panel,(key,label)) in enumerate(quantities)
        ax=latex_axis(fig[cld(panel,3),mod1(panel,3)],xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring(label),xscale=log10)
        values=getfield.(metrics,key); lines!(ax,x,values;color=series_color(panel),linewidth=2.5);scatter!(ax,x,values;color=:white,strokecolor=series_color(panel),strokewidth=1.4,markersize=9)
    end
    save_figure!(files,fig,output_dir,"ensemble_physical_scaling",formats;overwrite)
    phases=default_thermal_phases(); fig=publication_figure(size=(1280,520))
    for (panel,(key,label)) in enumerate(((:volume,"f_{\\mathrm{V}}"),(:mass,"f_{\\mathrm{M}}")))
        ax=latex_axis(fig[1,panel],xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring(label),xscale=log10)
        for p in eachindex(phases); values=[getfield(row,key)[p] for row in metrics];lines!(ax,x,values;color=series_color(p),linestyle=series_linestyle(p),linewidth=2.5,label=latex_legend_label(phases[p].name));scatter!(ax,x,values;color=series_color(p),marker=series_marker(p),markersize=8);end
        panel==1&&publication_legend!(ax)
    end
    save_figure!(files,fig,output_dir,"ensemble_phase_fractions",formats;overwrite)
    fig=publication_figure();ax=latex_axis(fig[1,1],xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring("\\zeta_2/\\zeta_3"),xscale=log10)
    for p in eachindex(phases); values=[row.zeta2[p] for row in metrics]; lines!(ax,x,values;color=series_color(p),linestyle=series_linestyle(p),linewidth=2.4,label=latex_legend_label(phases[p].name));scatter!(ax,x,values;color=series_color(p),marker=series_marker(p),markersize=9);end
    publication_legend!(ax);save_figure!(files,fig,output_dir,"ensemble_phase_zeta2",formats;overwrite)
    return files
end
