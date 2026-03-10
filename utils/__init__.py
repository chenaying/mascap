# -*- coding: utf-8 -*-
# 修复导入冲突：当utils既是文件又是包时，从utils.py导入函数
import sys
import os

# 获取utils.py文件的路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_utils_py_path = os.path.join(_parent_dir, 'utils.py')

# 如果utils.py存在，导入其中的函数
if os.path.exists(_utils_py_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils_py_module", _utils_py_path)
    utils_py_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_py_module)
    
    # 导出utils.py中的函数
    compose_discrete_prompts = utils_py_module.compose_discrete_prompts
    noise_injection = getattr(utils_py_module, 'noise_injection', None)
    parse_entities = getattr(utils_py_module, 'parse_entities', None)
    padding_captions = getattr(utils_py_module, 'padding_captions', None)
    entities_process = getattr(utils_py_module, 'entities_process', None)
else:
    raise ImportError(f"Cannot find utils.py at {_utils_py_path}")

# MeaCap相关导入（如果存在）
try:
    from .detect_utils import retrieve_concepts
    from .detect_utils_new import retrieve_concepts as retrieve_concepts_new
    from .parse_tool_new import parse, get_graph_dict, merge_graph_dict_new
except ImportError:
    # MeaCap模块不存在，忽略
    pass

__all__ = ['compose_discrete_prompts', 'noise_injection', 'parse_entities', 'padding_captions', 'entities_process']


