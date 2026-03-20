# Compile 

## Torch Compile

<div id="compile"></div>  

By the way, **cache-dit** is designed to work compatibly with <span style="color:hotpink;">**torch.compile.**</span> You can easily use cache-dit with torch.compile to further achieve a better performance. For example:

```python
cache_dit.enable_cache(pipe)

# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```
However, users intending to use **cache-dit** for DiT with **dynamic input shapes** should consider increasing the <span style="color:hotpink;">**recompile limit**</span> of `torch._dynamo`. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode. 
```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```  
Or, you can use the <span style="color:hotpink;">set_compile_configs</span> util func in cache-dit:   
```python
cache_dit.set_compile_configs()
```
