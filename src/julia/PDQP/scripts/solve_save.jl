import ArgParse
import GZip
import JSON3


import PDQP

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::PDQP.PdhgParameters,
    output_dir::String,
    instance_path::String,
    gpu_flag::Bool,
    pw,ss,niters,solve
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            qp = PDQP.qps_reader_to_standard_form(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        # @info "$(qp.objective_matrix)"
        # @info "$(qp.objective_constant)"
        # quit()
        @info "!!!!!! iters:  $(niters)"
        if niters >= 0
            if gpu_flag
                output = PDQP.optimize_gpu(parameters, qp)
            else
                filename = basename(instance_path)
                @info "primal weight: $(pw)     stepsize: $(ss)"
                if solve == 0
                    PDQP.write_model(parameters, qp,filename,ss,pw,niters)
                else
                    output,output_scaled = PDQP.optimize(parameters, qp,filename,ss,pw,niters)
                end
            end
        else
            if gpu_flag
                output = PDQP.optimize_gpu(parameters, qp)
            else
                if solve == 0
                    PDQP.write_model(parameters, qp,filename,ss,pw,niters)
                else
                    output,output_scaled = PDQP.optimize(parameters, qp,"",ss,pw,niters)
                end
            end
        end
        
        if solve==1
            log = PDQP.SolveLog()
            log.instance_name = instance_name
            log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
            log.termination_reason = output.termination_reason
            log.termination_string = output.termination_string
            log.iteration_count = output.iteration_count
            log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
            log.solution_stats = output.iteration_stats[end]
            log.solution_type = PDQP.POINT_TYPE_AVERAGE_ITERATE
        
            summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
            open(summary_output_path, "w") do io
                write(io, JSON3.write(log, allow_inf = true))
            end
        
            log.iteration_stats = output.iteration_stats
            full_log_output_path =
                joinpath(output_dir, instance_name * "_full_log.json.gz")
            GZip.open(full_log_output_path, "w") do io
                write(io, JSON3.write(log, allow_inf = true))
            end
        
            primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
            write_vector_to_file(primal_output_path, output.primal_solution)
        
            dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
            write_vector_to_file(dual_output_path, output.dual_solution)
            
            primal_output_path = joinpath(output_dir, instance_name * "_primal_scaled.txt")
            write_vector_to_file(primal_output_path, output_scaled.primal_solution)
        
            dual_output_path = joinpath(output_dir, instance_name * "_dual_scaled.txt")
            write_vector_to_file(dual_output_path, output_scaled.dual_solution)
        end
    end     

    inner_solve()
   
    return
end

function warm_up(qp::PDQP.QuadraticProgrammingProblem, gpu_flag::Bool,filename)
    restart_params = PDQP.construct_restart_parameters(
        PDQP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDQP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = PDQP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-3,
        eps_optimal_relative = 1.0e-3,
        time_sec_limit = Inf,
        iteration_limit = 10,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = PDQP.PdhgParameters(
        10,
        true,
        1.0,
        1.0,
        true,
        0,
        true,
        96,
        termination_params_warmup,
        restart_params,
        PDQP.ConstantStepsizeParams(),
    )
    if gpu_flag
        PDQP.optimize_gpu(params_warmup, qp);
    else
        PDQP.optimize(params_warmup, qp,filename);
    end
end

function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_path"
        help = "The path to the instance to solve in .mps.gz or .mps format."
        arg_type = String
        required = true

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        required = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-3

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 500.0

        "--use_gpu"
        help = "Using GPU: 0-false, 1-true"
        arg_type = Int64
        default = 0

        "--primal_weight"
        help = "initial primal weight, negative value for default"
        arg_type = Float64
        default = -1.0

        "--step_size"
        help = "initial step size, negative value for default"
        arg_type = Float64
        default = -1.0

        "--iteration"
        help = "initial iteration, 0 value for default"
        arg_type = Int64
        default = 0

        "--solve"
        help = "whether to solve problem"
        arg_type = Int64
        default = 1

        "--checkiter"
        help = "how frequent to check termination"
        arg_type = Int64
        default = 96
    end

    return ArgParse.parse_args(arg_parse)
end



function main()
    parsed_args = parse_command_line()
    instance_path = parsed_args["instance_path"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    output_directory = parsed_args["output_directory"]
    gpu_flag = Bool(parsed_args["use_gpu"])
    pw = parsed_args["primal_weight"]
    ss = parsed_args["step_size"]
    init_iter = parsed_args["iteration"]
    solve = parsed_args["solve"]
    checkiter = parsed_args["checkiter"]


    qp = PDQP.qps_reader_to_standard_form(instance_path)

    oldstd = stdout
    redirect_stdout(devnull)
    filename = basename(instance_path)
    # warm_up(qp, gpu_flag, "");
    redirect_stdout(oldstd)

    restart_params = PDQP.construct_restart_parameters(
        PDQP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        PDQP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.2,                    # primal_weight_update_smoothing
    )

    termination_params = PDQP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = PDQP.PdhgParameters(
        10,
        true,
        1.0,
        1.0,
        true,
        3,
        true,
        checkiter,
        termination_params,
        restart_params,
        PDQP.ConstantStepsizeParams(),  
    )

    if solve==0
        @info "Model output only"
    end

    solve_instance_and_output(
        params,
        output_directory,
        instance_path,
        gpu_flag,
        pw,
        ss,
        init_iter,
        solve
    )

    if solve==0
        @info "Finished model output"
    end

end

main()
