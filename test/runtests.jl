using CubeAnalysis
using FITSIO
using HDF5
using Test

function write_cube(path, cube)
    FITS(path, "w") do fits
        write(fits, cube)
    end
end

@testset "CubeAnalysis" begin
    @test !CubeAnalysis.save_data(Dict("run" => Dict()))
    style_figure = CubeAnalysis.Figure()
    style_axis = CubeAnalysis.latex_axis(style_figure[1, 1]; title="must be hidden", xlabel="x")
    @test isempty(style_axis.title[])
    @test !style_axis.xgridvisible[]
    @test !style_axis.ygridvisible[]
    @test !style_axis.xminorgridvisible[]
    @test !style_axis.yminorgridvisible[]
    @test !style_axis.rightspinevisible[]
    @test !style_axis.topspinevisible[]
    @test style_axis.xtickalign[] == 1
    @test style_axis.ytickalign[] == 1
    @test occursin("v_x", string(CubeAnalysis.latex_text("v_x")))
    @test !occursin(raw"v\_x", string(CubeAnalysis.latex_text("v_x")))
    @test occursin(raw"\log_{10}", string(CubeAnalysis.latex_text("log10 v_x")))
    log_figure = CubeAnalysis.Figure()
    native_log_axis = CubeAnalysis.latex_axis(log_figure[1, 1]; xscale=log10, yscale=log10)
    @test native_log_axis.xminorticksvisible[]
    @test native_log_axis.yminorticksvisible[]
    tick_values, tick_labels = CubeAnalysis.latex_log_ticks(0.1, 100.0)
    @test tick_values ≈ [0.1, 1.0, 10.0, 100.0]
    @test string.(tick_labels) == [raw"$10^{-1}$", raw"$10^{0}$",
        raw"$10^{1}$", raw"$10^{2}$"]
    explicit_log_axis = CubeAnalysis.latex_axis(log_figure[1, 2];
        xlogcoordinates=true, ylogcoordinates=true)
    @test explicit_log_axis.xminorticksvisible[]
    @test explicit_log_axis.yminorticksvisible[]
    @test !explicit_log_axis.xminorgridvisible[]
    @test !explicit_log_axis.yminorgridvisible[]
    @test CubeAnalysis.latex_layout_label(style_figure[0, 1], "must be hidden") === nothing
    equilibrium_temperature = 5000.0
    equilibrium_density = CubeAnalysis.thermal_equilibrium_density(equilibrium_temperature)
    cooling_over_heating = 1.0e7 * exp(-1.184e5 / (equilibrium_temperature + 1000.0)) +
        1.4e-2 * sqrt(equilibrium_temperature) * exp(-92.0 / equilibrium_temperature)
    @test equilibrium_density * cooling_over_heating ≈ 1.0
    equilibrium_n, equilibrium_t = CubeAnalysis.thermal_equilibrium_curve(0.01, 1000.0)
    @test length(equilibrium_n) == length(equilibrium_t) > 100
    @test all(n -> 0.01 <= n <= 1000.0, equilibrium_n)
    @test CubeAnalysis.phase_isotherm_colors(Dict(), 3) == [:yellow, :orange, :red]
    @test CubeAnalysis.phase_isotherm_colors(
        Dict("isotherm_colors" => ["red", "orange"]), 3) == [:red, :orange, :red]
    uniform_weights = ones(Float32, 4, 4, 4)
    vx_test = [Float32(i) for i in 1:4, _ in 1:4, _ in 1:4]
    zeros_test = zeros(Float32, 4, 4, 4)
    sigma_test, velocity_mean_test = CubeAnalysis.velocity_dispersion_1d(
        uniform_weights, vx_test, zeros_test, zeros_test)
    @test velocity_mean_test[1] ≈ 2.5
    @test sigma_test ≈ sqrt(1.25 / 3)
    aligned_density = [Float32(i) for i in 1:8, _ in 1:8, _ in 1:8]
    aligned_x = copy(aligned_density)
    aligned_zero = zeros(Float32, size(aligned_density))
    aligned_one = ones(Float32, size(aligned_density))
    aligned_result = CubeAnalysis.ibanez_xi_analysis(aligned_density,
        aligned_one, aligned_zero, aligned_zero,
        aligned_x, aligned_zero, aligned_zero;
        bins=4, density_range=(0.5, 10.0), minimum=1,
        blocks_per_axis=2, periodic=false)
    @test aligned_result.gradient_global.xi ≈ 1.0
    @test aligned_result.velocity_global.xi ≈ 1.0
    @test aligned_result.sigma_v > 0
    moderately_aligned_result = CubeAnalysis.ibanez_xi_analysis(aligned_density,
        fill(0.8f0, size(aligned_density)), fill(0.6f0, size(aligned_density)), aligned_zero,
        aligned_x, aligned_zero, aligned_zero;
        bins=4, density_range=(0.5, 10.0), minimum=1,
        blocks_per_axis=2, periodic=false)
    @test moderately_aligned_result.gradient_global.xi ≈ 1.0
    @test CubeAnalysis.xi_category(0.8, 0.75, 0.25) == 1
    @test CubeAnalysis.xi_category(0.3, 0.75, 0.25) == 0
    xi_rendered = mktempdir() do directory
        figure = CubeAnalysis.Figure(size=(500, 350))
        axis = CubeAnalysis.latex_axis(figure[1, 1]; xscale=log10)
        CubeAnalysis.plot_xi_curve!(axis, aligned_result.gradient, (0.5, 10.0))
        files = String[]
        CubeAnalysis.save_figure!(files, figure, directory, "xi_tick_regression", ["png"];
            overwrite=true)
        length(files) == 1 && isfile(only(files))
    end
    @test xi_rendered
    xi_existing_is_skipped = mktempdir() do directory
        existing = joinpath(directory, "xi_vs_sigma_v.png")
        write(existing, "preserve me")
        files = String[]
        result = CubeAnalysis.plot_xi_vs_sigma_v!(files, ["missing simulation"],
            directory, ["png"]; overwrite=false)
        result === files && isempty(files) && read(existing, String) == "preserve me"
    end
    @test xi_existing_is_skipped
    cube = reshape(Float32.(1:512), 8, 8, 8)
    gx, gy, gz = periodic_gradient(cube, 8.0)
    @test size(gx) == size(cube)
    @test size(gy) == size(cube)
    @test size(gz) == size(cube)
    k, power, modes = isotropic_spectrum(cube; bins=8)
    @test length(k) == length(power) == length(modes)
    @test all(power .>= 0)
    logk_test, logpower_test = CubeAnalysis.log_spectrum_coordinates([0.1, 1.0, 10.0],
        [100.0, 10.0, 1.0])
    @test logk_test ≈ [-1.0, 0.0, 1.0]
    @test logpower_test ≈ [2.0, 1.0, 0.0]

    @testset "analytic geometry and spectra" begin
        shape = (32, 24, 16)
        lengths = (2π, 4π, π)
        wave = Array{Float32}(undef, shape)
        for kindex in 1:shape[3], jindex in 1:shape[2], iindex in 1:shape[1]
            wave[iindex, jindex, kindex] = sin(2π * (iindex - 1) / shape[1]) +
                sin(2π * (jindex - 1) / shape[2]) + sin(2π * (kindex - 1) / shape[3])
        end
        gx, gy, gz = periodic_gradient(wave, lengths)
        @test gx[1, 1, 1] ≈ 1.0 atol = 0.02
        @test gy[1, 1, 1] ≈ 0.5 atol = 0.02
        @test gz[1, 1, 1] ≈ 2.0 atol = 0.06

        sinusoid = [Float32(sin(3 * 2π * (i - 1) / shape[1]))
            for i in 1:shape[1], _ in 1:shape[2], _ in 1:shape[3]]
        physical_k, physical_power, _ = isotropic_spectrum(sinusoid; bins=24,
            box_size=(2π, 2π, 2π))
        @test physical_k[argmax(physical_power)] ≈ 3.0 atol = 0.35

        zero_field = zeros(Float32, shape)
        vector_spectrum = CubeAnalysis.vector_spectrum_details(sinusoid, zero_field,
            zero_field; bins=24, box_size=(2π, 2π, 2π))
        peak = argmax(vector_spectrum.total)
        @test vector_spectrum.compressive[peak] / vector_spectrum.total[peak] > 0.99

        shear_spectrum = CubeAnalysis.vector_spectrum_details(zero_field, sinusoid,
            zero_field; bins=24, box_size=(2π, 2π, 2π))
        vorticity_spectrum = CubeAnalysis.vector_spectrum_details(zero_field, sinusoid,
            zero_field; bins=24, box_size=(2π, 2π, 2π), transform="curl")
        vorticity_peak = argmax(vorticity_spectrum.total)
        @test vorticity_spectrum.k[vorticity_peak] ≈ 3.0 atol = 0.35
        @test vorticity_spectrum.total[vorticity_peak] /
            shear_spectrum.total[vorticity_peak] ≈ 9.0 rtol = 1e-5
        @test vorticity_spectrum.compressive[vorticity_peak] ≤
            1e-6 * vorticity_spectrum.total[vorticity_peak]
        @test CubeAnalysis.nyquist_wavenumber(shape, lengths) ≈ 6.0
        @test CubeAnalysis.fundamental_wavenumber(lengths) ≈ 0.5
        @test CubeAnalysis.injection_wavenumber(lengths; modes=4.0) ≈ 2.0
        @test CubeAnalysis.dissipation_wavenumber(shape, lengths; cells=4.0) ≈ 3.0
        @test CubeAnalysis.inertial_fit_range([1.0, 5.0], shape, lengths,
            ("x", "y", "z"); injection_modes=4.0, dissipation_cells=4.0) ≈ [2.0, 3.0]
        fit_k = [1.0, 2.0, 2.5, 3.0, 4.0]
        fit_power = [1.0e8, 2.0^-2, 2.5^-2, 3.0^-2, 1.0e8]
        inertial_slope, _, inertial_points = CubeAnalysis.spectral_fit(
            fit_k, fit_power, [2.0, 3.0])
        @test inertial_slope ≈ -2.0
        @test inertial_points == 3
        @test_throws ErrorException CubeAnalysis.injection_wavenumber(lengths; modes=0.5)
        @test_throws ErrorException CubeAnalysis.dissipation_wavenumber(shape, lengths;
            cells=1.5)
        reference_x, reference_y = CubeAnalysis.reference_power_law(
            [1.0, 2.0, 4.0, 8.0], ones(4), -5 / 3)
        @test (reference_y[end] - reference_y[1]) /
            (reference_x[end] - reference_x[1]) ≈ -5 / 3

        cross = CubeAnalysis.cross_spectrum_details(sinusoid, 2f0 .* sinusoid;
            bins=24, box_size=(2π, 2π, 2π))
        @test cross.coherence[argmax(abs.(cross.cross))] ≈ 1.0 atol = 1e-5
        lags, orders, structure, samples = CubeAnalysis.structure_functions(sinusoid;
            lags=[1, 2], orders=[1, 2, 3], samples_per_lag=2_000, seed=8)
        @test size(structure) == (2, 3)
        @test all(samples .> 0)
        isotropic_structure_cube = zeros(Float32, 32, 32, 32)
        separation, structure_dissipation, structure_injection, separation_label =
            CubeAnalysis.structure_plot_scales(isotropic_structure_cube, [1, 2, 4],
                (2π, 2π, 2π), ("x", "y", "z"); injection_modes=4.0,
                dissipation_cells=4.0, box_unit="pc")
        @test separation ≈ [2π / 32, 4π / 32, 8π / 32]
        @test structure_dissipation ≈ 8π / 32
        @test structure_injection ≈ π / 2
        @test occursin("pc", string(separation_label))
        @test CubeAnalysis.kolmogorov_ess(3) ≈ 1.0
        @test CubeAnalysis.she_leveque_ess(3) ≈ 1.0
        @test CubeAnalysis.boldyrev_ess(3) ≈ 1.0
        phase_lags, phase_orders, phase_structure, phase_counts =
            CubeAnalysis.phase_velocity_structure(sinusoid, zero_field, zero_field,
                fill(8000.0f0, size(sinusoid)), CubeAnalysis.default_thermal_phases();
                lags=[1, 2], orders=[1, 2, 3], samples_per_lag=2_000, seed=9)
        @test phase_lags == [1, 2]
        @test phase_orders == [1, 2, 3]
        @test size(phase_structure) == (2, 3, 3)
        @test all(phase_counts[:, 3] .> 0)
        radius, correlation, _, _ = CubeAnalysis.radial_autocorrelation(sinusoid;
            bins=12, box_size=(2π, 2π, 2π))
        @test !isempty(radius)
        @test correlation[1] <= 1.0 + 1e-5

        anisotropic = CubeAnalysis.anisotropic_spectrum(sinusoid, (1.0, 0.0, 0.0);
            parallel_bins=12, perpendicular_bins=12, radial_bins=12,
            box_size=(2π, 2π, 2π))
        @test sum(anisotropic.parallel) > 100 * sum(anisotropic.perpendicular)

        directional = CubeAnalysis.directional_structure_functions(
            [sinusoid, zero_field, zero_field]; lags=[1, 2], orders=[2, 3, 4],
            box_size=(2π, 2π, 2π))
        x_rows = filter(row -> row.axis == "x", directional)
        @test all(row -> row.longitudinal[1] > 0, x_rows)
        @test all(row -> row.transverse[1] == 0, x_rows)

        uniform_density = fill(2.0f0, 4, 5, 10)
        column_cfg = Dict("grid" => Dict("box_size" => [8.0, 10.0, 50.0],
            "axis_order" => ["x", "y", "z"], "box_unit" => "pc"))
        expected_column = 2.0 * 50.0 * CubeAnalysis.length_unit_to_cm("pc")
        @test all(CubeAnalysis.column_density_map(uniform_density, "z", column_cfg) .≈
            expected_column)
        @test size(CubeAnalysis.column_density_map(uniform_density, "x", column_cfg)) == (5, 10)
        @test CubeAnalysis.column_density_unit("cm^-3") == "cm^-2"

        excursion = zeros(Float32, 4, 4, 4)
        excursion[1, 1, 1] = 1; excursion[4, 4, 4] = 1
        topology = CubeAnalysis.excursion_metrics(excursion, 0.5; periodic=false)
        @test topology.volume_fraction == 2 / 64
        @test topology.components == 2
        @test topology.largest_fraction == 0.5
    end

    @testset "HRO definitions" begin
        scalar = [Float32(i) for i in 1:8, _ in 1:8, _ in 1:8]
        condition = @. 1f0 + scalar / 100
        ones_field = ones(Float32, size(scalar)); zeros_field = zero(ones_field)
        parallel = CubeAnalysis.alignment_analysis(scalar, ones_field, zeros_field,
            zeros_field, condition; periodic=false, weighting="uniform", minimum=2)
        perpendicular = CubeAnalysis.alignment_analysis(scalar, zeros_field, ones_field,
            zeros_field, condition; periodic=false, weighting="uniform", minimum=2)
        @test all(value -> value ≈ 1, filter(isfinite, parallel[4]))
        @test all(value -> value ≈ -1, filter(isfinite, perpendicular[4]))
    end

    mktempdir() do directory
        input_dir = joinpath(directory, "input")
        output_dir = joinpath(directory, "output")
        mkpath(input_dir)
        density = @. 0.1f0 + cube / 100
        temperature = @. 100f0 + cube
        bx = @. sin(cube / 20)
        by = @. cos(cube / 25)
        bz = fill(0.5f0, size(cube))
        for (name, values) in (
            "density" => density, "temperature" => temperature,
            "Bx" => bx, "By" => by, "Bz" => bz,
        )
            write_cube(joinpath(input_dir, "$name.fits"), values)
        end

        config = joinpath(directory, "test.toml")
        open(config, "w") do io
            print(io, """
            [run]
            input_dir = "$input_dir"
            output_dir = "$output_dir"
            formats = ["png"]
            sample_stride = 2
            atomic_directory = true
            save_data = true

            [plots]
            summary = true
            slices = true
            projections = true
            histograms = true
            phase_diagrams = true
            relations = true
            power_spectra = true
            alignments = true

            [slices]
            axes = ["z"]
            positions = [0.5]

            [projections]
            axes = ["z"]
            statistics = ["mean"]

            [histograms]
            bins = 12

            [[histograms.groups]]
            name = "magnetic_components"
            fields = ["Bx", "By", "Bz"]

            [spectra]
            bins = 8
            fields = ["density", "Bmag"]

            [fields.density]
            file = "density.fits"
            log = true

            [fields.temperature]
            file = "temperature.fits"
            log = true

            [fields.Bx]
            file = "Bx.fits"

            [fields.By]
            file = "By.fits"

            [fields.Bz]
            file = "Bz.fits"

            [[derived.vectors]]
            name = "Bmag"
            components = ["Bx", "By", "Bz"]
            log = true

            [[phase_diagrams]]
            name = "n_T"
            x = "density"
            y = "temperature"
            bins = 12
            log_x = true
            log_y = true
            thermal_overlay = "temperature"
            isotherms = [100.0, 1000.0, 8000.0]

            [[relations]]
            name = "B_n"
            x = "density"
            y = "Bmag"
            bins = 6
            minimum_per_bin = 2
            log_x = true
            log_y = true

            [[alignments]]
            name = "grad_n_B"
            scalar = "density"
            vector = ["Bx", "By", "Bz"]
            condition_bins = 5
            angle_bins = 8
            minimum_per_bin = 2
            """)
        end

        result = run_analysis(config)
        @test result.output_dir == output_dir
        @test isfile(joinpath(output_dir, "cube_summary.csv"))
        @test isfile(joinpath(output_dir, "phase_n_t.png"))
        @test isfile(joinpath(output_dir, "histogram_magnetic_components.png"))
        @test isfile(joinpath(output_dir, "spectrum_density.csv"))
        @test isfile(joinpath(output_dir, "alignment_grad_n_b.csv"))
        @test isfile(joinpath(output_dir, "analysis_manifest.toml"))
        @test length(result.files) >= 10

        hdf5_path = joinpath(input_dir, "simulation.h5")
        HDF5.h5open(hdf5_path, "w") do file
            file["/fields/density"] = permutedims(density, (3, 2, 1))
            file["/fields/temperature"] = temperature
        end
        hdf5_output = joinpath(directory, "hdf5_output")
        hdf5_config = joinpath(directory, "hdf5.toml")
        open(hdf5_config, "w") do io
            print(io, """
            [run]
            input_dir = "$input_dir"
            output_dir = "$hdf5_output"
            field_by_field = true
            memory_budget_gb = 0.01
            save_data = true

            [plots]
            summary = true

            [fields.density]
            file = "simulation.h5"
            dataset = "/fields/density"
            permutation = [3, 2, 1]
            scale = 2.0

            [fields.temperature]
            file = "simulation.h5"
            dataset = "/fields/temperature"
            """)
        end
        hdf5_cfg = load_config(hdf5_config)
        hdf5_fields, _ = load_fields(hdf5_cfg)
        @test hdf5_fields isa CubeAnalysis.LazyFieldStore
        @test hdf5_fields["density"] == 2f0 .* density
        @test hdf5_fields["temperature"] == temperature
        hdf5_result = run_analysis(hdf5_config)
        @test isfile(joinpath(hdf5_result.output_dir, "cube_summary.csv"))

        series_output = joinpath(directory, "series_output")
        series_config = joinpath(directory, "series.toml")
        open(series_config, "w") do io
            print(io, """
            [run]
            input_dir = "$input_dir"
            output_dir = "$series_output"
            field_by_field = true
            save_data = true

            [plots]
            summary = true

            [fields.density]
            file = "density.fits"

            [[snapshots]]
            name = "first"
            time = 0.0
            input_dir = "$input_dir"

            [[snapshots]]
            name = "second"
            time = 1.0
            input_dir = "$input_dir"
            """)
        end
        series_result = run_analysis(series_config)
        @test isfile(joinpath(series_result.output_dir, "snapshot_evolution.csv"))
        @test isfile(joinpath(series_result.output_dir, "first", "cube_summary.csv"))
        @test length(readlines(joinpath(series_result.output_dir, "snapshot_evolution.csv"))) == 3
    end
end
