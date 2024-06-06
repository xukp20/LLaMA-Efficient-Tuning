# 0419 coloop phi-2
# def load_custom_model():
#     modeling_path = "/cephfs/xukangping/code/experiments/loop/modeling_coloop_phi.py"
#     import sys, os
#     sys.path.append(os.path.dirname(modeling_path))
#     from modeling_coloop_phi import CoLoopPhiForCasualLM, CoLoopPhiConfig
#     from transformers import AutoTokenizer
#     model_path = "/cephfs/shared/hf_cache/hub/models--microsoft--phi-2/snapshots/710686f446f02286c858c11f052acb87c306ddd2"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     from transformers.models.phi.modeling_phi import PhiForCausalLM
    
#     fix_phi = "0"
#     print(f"FIX_PHI: {fix_phi}")
    
#     base_phi = PhiForCausalLM.from_pretrained(model_path, torch_dtype="auto")

#     coloop_phi_config = CoLoopPhiConfig(
#         gate_type="stop",
#         loop_in_layer=8,
#         loop_out_layer=-8,
#         stop_i=2,
#         passing_threshold=0.5,
#         weight_strategy="final",
#         **base_phi.config.to_dict()
#     )

    # coloop_phi_config["architectures"] = [
    #     "CoLoopPhiForCasualLM",
    #     "CoLoopPhiModel"
    # ]

    # coloop_phi_config["auto_map"] = {
    #     "AutoConfig": "modeling_coloop_phi.CoLoopPhiConfig",
    #     "AutoModel": "modeling_coloop_phi.CoLoopPhiModel",
    #     "AutoModelForCausalLM": "modeling_coloop_phi.CoLoopPhiForCasualLM"
    # }

#     print(coloop_phi_config)

#     coloop_phi_model = CoLoopPhiForCasualLM.from_phi(base_phi, coloop_phi_config)

#     if fix_phi == "1":
#         coloop_phi_model.fix_phi()
#     else:
#         coloop_phi_model.train_all()

#     return coloop_phi_model, coloop_phi_config


# 0503 coloop  phi-3
# def load_custom_model():
#     modeling_path = "/cephfs/xukangping/code/experiments/loop/modeling_coloop_phi3.py"
#     import sys, os
#     sys.path.append(os.path.dirname(modeling_path))
#     from modeling_coloop_phi3 import CoLoopPhi3ForCasualLM, CoLoopPhi3Config, Phi3ForCausalLM
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     model_path = "/cephfs/shared/hf_cache/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/653ee820c4f2ee66427e997b4a8ca3e9323e7d46"
#     # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
#     fix_phi = "0"
#     print(f"FIX_PHI: {fix_phi}")
    
#     base_phi = Phi3ForCausalLM.from_pretrained(model_path, torch_dtype="auto")

#     coloop_phi_config = CoLoopPhi3Config(
#         gate_type="stop",
#         loop_in_layer=8,
#         loop_out_layer=-8,
#         stop_i=1,
#         passing_threshold=0.5,
#         weight_strategy="final",
#         **base_phi.config.to_dict()
#     )

#     # coloop_phi_config["architectures"] = [
#     #     "CoLoopPhi3ForCasualLM",
#     #     "CoLoopPhi3Model"
#     # ]

#     # coloop_phi_config["auto_map"] = {
#     #     "AutoConfig": "modeling_coloop_phi3.CoLoopPhi3Config",
#     #     "AutoModel": "modeling_coloop_phi3.CoLoopPhi3Model",
#     #     "AutoModelForCausalLM": "modeling_coloop_phi3.CoLoopPhi3ForCasualLM"
#     # }

#     print(coloop_phi_config)

#     coloop_phi_model = CoLoopPhi3ForCasualLM.from_phi(base_phi, coloop_phi_config)

#     # if fix_phi == "1":
#     #     coloop_phi_model.fix_phi()
#     # else:
#     #     coloop_phi_model.train_all()

#     return coloop_phi_model, coloop_phi_config

# 0509, load uni phi
# def load_custom_model():
    # pass


# def load_custom_model():
#     modeling_path = "/cephfs/xukangping/code/experiments/loop/modeling_coloop_phi.py"
#     import sys, os
#     sys.path.append(os.path.dirname(modeling_path))
#     from modeling_coloop_phi import CoLoopPhiForCasualLM, CoLoopPhiConfig
#     from transformers import AutoTokenizer
#     model_path = "/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-21-21-15/checkpoint-6000"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     from transformers.models.phi.modeling_phi import PhiForCausalLM
#     base_phi = PhiForCausalLM.from_pretrained(model_path, torch_dtype="auto")

#     coloop_phi_config = CoLoopPhiConfig(
#         gate_type="temperature",
#         loop_in_layer=8,
#         loop_out_layer=-8,
#         stop_i=1,
#         passing_threshold=0.5,
#         weight_strategy="final",
#         **base_phi.config.to_dict()
#     )

#     coloop_phi_model = CoLoopPhiForCasualLM.from_phi(base_phi, coloop_phi_config)

#     coloop_phi_model.fix_phi()

#     return coloop_phi_model, coloop_phi_config


# import os
# os.environ["FIX_PHI"] = "1"
# load_custom_model()

# 0520 load coloop uni phi, mlp version
# def load_custom_model():
#     import os
#     modeling_path = "/cephfs/xukangping/root/models/uni-coloop-phi-0520"
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained(modeling_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(modeling_path, torch_dtype="auto", trust_remote_code=True)

#     # fix phi
#     fix_phi = os.environ.get("FIX_PHI", "0")
#     print(f"FIX_PHI: {fix_phi}")
#     if fix_phi == "1":
#         model.fix_phi()
#     else:
#         model.train_all()

    # return model, model.config

# 0602 load local loop phi

def get_layers_from_str(layers_str, total_layers):
    if layers_str == 'all':
        loop_layers = list(range(total_layers))
    else:
        layers = layers_str.split(',')
        loop_layers = []
        for layer in layers:
            if '-' in layer:
                start, end = map(int, layer.split('-'))
                loop_layers.extend(list(range(start, end)))
            else:
                loop_layers.append(int(layer))
            
    loop_layers = sorted(list(set(loop_layers)))
    return loop_layers

def load_loop_class(modeling_code_path, model_type):
    import sys
    sys.path.append(modeling_code_path)
    if model_type == 'phi-2':
        module_name = 'modeling_local_loop_phi'
        config_class_name = 'LocalLoopPhiConfig'
        model_class_name = 'LocalLoopPhiForCausalLM'
        load_func_name = 'from_base'
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    module = __import__(module_name)
    config_class = getattr(module, config_class_name)
    model_class = getattr(module, model_class_name)
    load_func = getattr(model_class, load_func_name)

    auto_map = {
        "AutoConfig": f"{module_name}.{config_class_name}",
        "AutoModel": f"{module_name}.{model_class_name}",
        "AutoModelForCausalLM": f"{module_name}.{model_class_name}",
    }

    return config_class, model_class, load_func, auto_map, module_name


def load_custom_model():
    import os
    # load MODEL_PATH, PORJECTION_TYPE, LOOP_LAYERS, LOOP_TIMES, FIX_PROJECTION, PROJECTION_INIT from os.environ
    model_path = os.getenv("MODEL_PATH")
    projection_type = os.getenv("PROJECTION_TYPE")
    loop_layers = os.getenv("LOOP_LAYERS")
    loop_times = int(os.getenv("LOOP_TIMES"))
    fix_projection = os.getenv("FIX_PROJECTION")
    projection_init = os.getenv("PROJECTION_INIT")
    fix_llm = os.getenv("FIX_LLM")
    base_model = os.getenv("BASE_MODEL")

    print(f"MODEL_PATH: {model_path}")
    print(f"PROJECTION_TYPE: {projection_type}")
    print(f"LOOP_LAYERS: {loop_layers}")
    print(f"LOOP_TIMES: {loop_times}")
    print(f"FIX_PROJECTION: {fix_projection}")
    print(f"PROJECTION_INIT: {projection_init}")
    print(f"FIX_LLM: {fix_llm}")
    print(f"BASE_MODEL: {base_model}")
    
    fix_projection = True if str(fix_projection) == "1" else False
    fix_llm = True if str(fix_llm) == "1" else False
    base_model = True if str(base_model) == "1" else False

    if base_model:
        modeling_code_path = "/cephfs/xukangping/code/experiments/local_loop/models"
        model_type = "phi-2"
        config_class, model_class, from_model, auto_map, module_name = load_loop_class(modeling_code_path, model_type)

        # load base config from model_path
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        base_config = AutoConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")

        loop_config = config_class(
            projection_type=projection_type,
            loop_layers=get_layers_from_str(str(loop_layers), base_config.num_hidden_layers),
            loop_times=loop_times,
            fix_projection=fix_projection,
            projection_init=projection_init,
            **base_config.to_dict()
        )
        loop_config_dict = loop_config.to_dict()
        loop_config_dict.update({
            "auto_map": auto_map,
        })
        loop_config = config_class.from_dict(loop_config_dict)

        # load model
        loop_model = from_model(base_model, loop_config).to(base_model.dtype)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        loop_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
        loop_config = AutoConfig.from_pretrained(model_path)
        
    # set all parameters to be fixed
    for param in loop_model.parameters():
        param.requires_grad = not fix_llm
    
    loop_model.fix_projection(fix_projection)

    # print the trainable parameters
    if fix_llm:
        print("Trainable parameters:")
        for name, param in loop_model.named_parameters():
            if param.requires_grad:
                print(name)
    elif fix_projection:
        print("Not trainable parameters:")
        for name, param in loop_model.named_parameters():
            if not param.requires_grad:
                print(name)


    return loop_model, loop_config