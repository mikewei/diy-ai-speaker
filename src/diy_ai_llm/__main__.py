import sys
import runpy
import click

@click.command()
@click.option('--model', type=str, default='Qwen/Qwen3-14B-AWQ', help='Model path')
@click.option('--served-model-name', type=str, default='qwen3-14b-diy', help='Served model name')
@click.option('--gpu-memory-utilization', type=float, default=0.55, help='GPU memory utilization')
@click.option('--max-model-len', type=int, default=4096, help='Max sequence length')
@click.option('--max-num-seqs', type=int, default=2, help='Max number of sequences')
@click.option('--host', type=str, default='0.0.0.0', help='Host')
@click.option('--port', type=int, default=8000, help='Port')
def vllm_start(**kwargs):
    ctx = click.get_current_context()
    args_with_defaults = []
    for param in ctx.command.params:
        assert param.name is not None
        value = ctx.params.get(param.name)
        if isinstance(param, click.Option):
            # 构造 --key=value 形式
            if isinstance(value, bool):
                if value:
                    args_with_defaults.append(f"--{param.name.replace('_', '-')}")
            else:
                args_with_defaults.append(f"--{param.name.replace('_', '-')}={value}")
        elif isinstance(param, click.Argument):
            args_with_defaults.append(str(value))
    sys.argv = ['vllm.entrypoints.openai.api_server'] + args_with_defaults
    print(' '.join(sys.argv))
    runpy.run_module(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    vllm_start()