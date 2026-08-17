function weighted_moment_maps(density, velocity, dimension)
    size(density)==size(velocity) || error("Density and velocity cubes must have equal sizes")
    weight=Float64.(density); speed=Float64.(velocity)
    @. weight=ifelse(isfinite(weight)&&weight>0&&isfinite(speed),weight,0.0)
    denominator=dropdims(sum(weight;dims=dimension);dims=dimension)
    first=dropdims(sum(weight.*speed;dims=dimension);dims=dimension)./max.(denominator,eps())
    second=dropdims(sum(weight.*speed.^2;dims=dimension);dims=dimension)./max.(denominator,eps())
    dispersion=sqrt.(max.(second.-first.^2,0.0))
    return first,dispersion,denominator
end

function ppv_cube(density,velocity,dimension;channels=64,velocity_limits=nothing)
    finite=filter(isfinite,Float64.(velocity)); isempty(finite)&&error("PPV cube needs finite velocities")
    limits=isnothing(velocity_limits) ? quantile(finite,[0.005,0.995]) : Float64.(velocity_limits)
    limits[2]>limits[1] || (limits=[limits[1]-0.5,limits[1]+0.5])
    remaining=filter(!=(dimension),1:3); output=zeros(Float32,size(density,remaining[1]),size(density,remaining[2]),channels)
    edges=collect(range(limits[1],limits[2];length=channels+1))
    for index in CartesianIndices(density)
        rho=Float64(density[index]); v=Float64(velocity[index]); isfinite(rho+v)&&rho>0 || continue
        bin=searchsortedlast(edges,v); 1<=bin<=channels || continue
        output[index[remaining[1]],index[remaining[2]],bin]+=Float32(rho)
    end
    return output,0.5.*(edges[1:end-1].+edges[2:end])
end

function polarization_maps(density,b1,b2,dimension)
    weight=max.(Float64.(density),0.0); denominator=Float64.(b1).^2 .+ Float64.(b2).^2
    valid=isfinite.(weight) .& isfinite.(denominator) .& (denominator .> 0)
    q=dropdims(sum(ifelse.(valid,weight .* (Float64.(b1).^2 .- Float64.(b2).^2) ./ max.(denominator,eps()),0.0);dims=dimension);dims=dimension)
    u=dropdims(sum(ifelse.(valid,weight .* 2 .* Float64.(b1) .* Float64.(b2) ./ max.(denominator,eps()),0.0);dims=dimension);dims=dimension)
    intensity=dropdims(sum(ifelse.(valid,weight,0.0);dims=dimension);dims=dimension)
    return sqrt.(q.^2 .+ u.^2) ./ max.(intensity,eps()),0.5 .* atan.(u,q),intensity
end

function linewidth_size(density,vx,vy,vz;sizes=(4,8,16,32,64))
    scales=Int[]; dispersions=Float64[]
    for width in Int.(sizes)
        width<=minimum(size(density)) || continue; samples=Float64[]
        for k in 1:width:size(density,3)-width+1,j in 1:width:size(density,2)-width+1,i in 1:width:size(density,1)-width+1
            range=(i:i+width-1,j:j+width-1,k:k+width-1); w=max.(Float64.(@view density[range...]),0.0); total=sum(w); total>0||continue
            variance=0.0
            for component in (vx,vy,vz)
                values=Float64.(@view component[range...]); center=sum(w.*values)/total
                variance+=sum(w.*(values.-center).^2)/total
            end
            push!(samples,sqrt(variance/3))
        end
        isempty(samples)|| (push!(scales,width);push!(dispersions,median(samples)))
    end
    return scales,dispersions
end

function plot_synthetic_observations!(files,fields,metadata,cfg,output_dir,formats,overwrite)
    spec=get(cfg,"synthetic_observations",Dict()); density_name=string(get(spec,"density","density"))
    velocity_names=string.(get(spec,"velocity",["Vx","Vy","Vz"])); magnetic_names=string.(get(spec,"magnetic",["Bx","By","Bz"]))
    all(haskey(fields,n) for n in [density_name;velocity_names;magnetic_names])||error("Synthetic observations reference missing fields")
    density=fields[density_name]; grid=get(cfg,"grid",Dict()); axes_names=string.(get(spec,"lines_of_sight",["x","y","z"])); order=string.(get(grid,"axis_order",["x","y","z"])); unit=string(get(grid,"box_unit","pc")); column_maps=Matrix{Float64}[]
    for los in axes_names
        dimension=findfirst(==(los),order); isnothing(dimension)&&error("Unknown line of sight '$los'")
        centroid,dispersion,column=weighted_moment_maps(density,fields[velocity_names[dimension]],dimension)
        push!(column_maps,column)
        ppv,channels=ppv_cube(density,fields[velocity_names[dimension]],dimension;channels=Int(get(spec,"velocity_channels",64)))
        fig=publication_figure(size=(1320,500))
        for (panel,(map,label,cmap)) in enumerate(((centroid,"v_{\\mathrm{c}}\\ [\\mathrm{km\\,s^{-1}}]",:balance),(dispersion,"\\sigma_{v,\\mathrm{los}}\\ [\\mathrm{km\\,s^{-1}}]",:viridis)))
            ax=latex_axis(fig[1,2panel-1],xlabel=latexstring(order[filter(!=(dimension),1:3)[1]],"\\ [\\mathrm{",unit,"]}"),ylabel=latexstring(order[filter(!=(dimension),1:3)[2]],"\\ [\\mathrm{",unit,"]}"))
            hm=heatmap!(ax,map;colormap=cmap); latex_colorbar(fig[1,2panel],hm;label=latexstring(label))
        end
        save_figure!(files,fig,output_dir,"synthetic_moments_$(los)",formats;overwrite)

        profile=vec(sum(ppv;dims=(1,2))); selected=unique(round.(Int,range(1,length(channels);length=3)))
        fig=publication_figure(size=(1320,430)); ax=latex_axis(fig[1,1],xlabel=latexstring("v_{\\mathrm{los}}\\ [\\mathrm{km\\,s^{-1}}]"),ylabel=latexstring("I(v_{\\mathrm{los}})"))
        lines!(ax,channels,profile;color=PLOT_BLUE,linewidth=2.7)
        for (panel,channel) in enumerate(selected)
            mapaxis=latex_axis(fig[1,panel+1],xlabel=latexstring("x_{\\mathrm{pos}}"),ylabel=latexstring("y_{\\mathrm{pos}}"))
            heatmap!(mapaxis,ppv[:,:,channel];colormap=:magma)
        end
        save_figure!(files,fig,output_dir,"synthetic_ppv_$(los)",formats;overwrite)

        pos=filter(!=(dimension),1:3); polarization,angle,intensity=polarization_maps(density,fields[magnetic_names[pos[1]]],fields[magnetic_names[pos[2]]],dimension)
        fig=publication_figure(size=(1120,500)); ax1=latex_axis(fig[1,1],xlabel=latexstring("x_{\\mathrm{pos}}"),ylabel=latexstring("y_{\\mathrm{pos}}")); hm1=heatmap!(ax1,log10.(max.(intensity,eps()));colormap=:magma); latex_colorbar(fig[1,2],hm1;label=latexstring("\\log_{10}N"))
        step=max(1,size(angle,1)÷24)
        for j in 1:step:size(angle,2),i in 1:step:size(angle,1)
            a=angle[i,j]; len=0.42step; lines!(ax1,[i-len*cos(a),i+len*cos(a)],[j-len*sin(a),j+len*sin(a)];color=:white,linewidth=1.1)
        end
        ax2=latex_axis(fig[1,3],xlabel=latexstring("x_{\\mathrm{pos}}"),ylabel=latexstring("y_{\\mathrm{pos}}")); hm2=heatmap!(ax2,polarization;colormap=:viridis,colorrange=(0,1)); latex_colorbar(fig[1,4],hm2;label=latexstring("p"))
        save_figure!(files,fig,output_dir,"synthetic_polarization_$(los)",formats;overwrite)
    end
    fig=publication_figure(); ax=latex_axis(fig[1,1],xlabel=latexstring("\\log_{10}(N/\\langle N\\rangle)"),ylabel=latexstring("p[\\log_{10}(N/\\langle N\\rangle)]"),yscale=log10)
    for (index,column) in enumerate(column_maps)
        values=filter(isfinite,vec(column)); values=values[values.>0]; isempty(values)&&continue
        normalized=log10.(values./mean(values)); edges=collect(range(quantile(normalized,0.005),quantile(normalized,0.995);length=61)); counts=zeros(Float64,60)
        for value in normalized; bin=searchsortedlast(edges,value); 1<=bin<=60&&(counts[bin]+=1); end
        centers=0.5.*(edges[1:end-1].+edges[2:end]); counts./=max(sum(counts)*(edges[2]-edges[1]),eps())
        lines!(ax,centers,counts;color=series_color(index),linestyle=series_linestyle(index),linewidth=2.6,label=latex_legend_label("LOS "*axes_names[index]))
    end
    publication_legend!(ax); save_figure!(files,fig,output_dir,"synthetic_column_density_pdf",formats;overwrite)
    sizes,sigma=linewidth_size(density,(fields[n] for n in velocity_names)...;sizes=get(spec,"linewidth_sizes",[4,8,16,32,64]))
    spacing=mean(grid_spacings(density,get(grid,"box_size",1.0);axis_order=order)); scale=sizes.*spacing
    fig=publication_figure(); ax=latex_axis(fig[1,1],xlabel=latexstring("\\ell\\ [\\mathrm{",unit,"]}"),ylabel=latexstring("\\sigma_v(\\ell)\\ [\\mathrm{km\\,s^{-1}}]"),xscale=log10,yscale=log10)
    lines!(ax,scale,sigma;color=PLOT_BLUE,linewidth=2.7);scatter!(ax,scale,sigma;color=:white,strokecolor=PLOT_BLUE,strokewidth=1.4,markersize=9)
    save_figure!(files,fig,output_dir,"synthetic_linewidth_size",formats;overwrite)
    return files
end
