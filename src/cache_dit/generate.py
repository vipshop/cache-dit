# Import from the defining module: `cache_dit.cli.entrypoint` is now also a submodule name, which
# can shadow the function re-exported by the package depending on import order.
from .cli.generate import entrypoint


def main():
  entrypoint()


if __name__ == "__main__":
  main()
