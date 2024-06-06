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
        module_name = 'modeling_local_loop_phi_v2'
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
    update_style = os.getenv("UPDATE_STYLE")
    loop_layers = os.getenv("LOOP_LAYERS")
    loop_times = os.getenv("LOOP_TIMES")

    modeling_code_path = "/cephfs/xukangping/code/experiments/local_loop/models"
    model_type = "phi-2"
    config_class, model_class, from_model, auto_map, module_name = load_loop_class(modeling_code_path, model_type)

    # load base config from model_path
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    base_config = AutoConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")

    loop_config = config_class(
        update_style=update_style,
        loop_layers=get_layers_from_str(loop_layers, base_config.num_hidden_layers),
        loop_times=int(loop_times),
        **base_config.to_dict()
    )
    loop_config_dict = loop_config.to_dict()
    loop_config_dict.update({
        "auto_map": auto_map,
    })
    loop_config = config_class.from_dict(loop_config_dict)

    # load model
    loop_model = from_model(base_model, loop_config).to(base_model.dtype)

    return loop_model, loop_config