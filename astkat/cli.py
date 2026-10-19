"""
The astkat command: one subcommand per pipeline step.

    astkat init -i IMAGE_DIR -n NAME
    astkat extract
    astkat filter
    astkat match
    astkat fit -y
    astkat offsets
    astkat target TRAJECTORY_FILE
    astkat simulate --mode flat -i 100
    astkat stack

Every step runs in the project directory, where astkat init writes astkat_files/astkat.yaml.
"""

import importlib
import sys

# subcommand: (module, entry function, one-line help)
COMMANDS = {
    'init':     ('initialise', 'initialise_project', 'Create a project: directories, astkat.yaml and the deep image'),
    'extract':  ('extraction', 'extract_sources_from_images', 'Find sources in every epoch with PyBDSF'),
    'filter':   ('filtering', 'main', 'Keep point sources'),
    'match':    ('matching', 'main', 'Cross-match sources between epochs'),
    'fit':      ('fit', 'main', 'Joint astrometric fit'),
    'offsets':  ('offsets', 'main', 'Translations for every epoch'),
    'target':   ('target', 'main', 'Correct a target\'s positions with the epoch translations'),
    'simulate': ('simulate', 'main', 'Simulate epoch images or a moving target with known errors'),
    'stack':    ('stacking', 'main', 'Rebuild the stacked deep image of an existing project'),
}


def usage():
    from . import __version__
    lines = [f"AstKAT {__version__}", "", "usage: astkat <command> [options]", "", "commands:"]
    lines += [f"  {name:<10}{help_text}" for name, (_, _, help_text) in COMMANDS.items()]
    lines += ["", "astkat <command> -h shows a command's options."]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(usage())
        return 0
    if sys.argv[1] in ('-V', '--version'):
        from . import __version__
        print(__version__)
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"astkat: unknown command '{command}'\n\n{usage()}", file=sys.stderr)
        return 2
    module_name, function_name, _ = COMMANDS[command]
    # The step's own argparse sees "astkat <command>" as the program and the rest as its options
    sys.argv = [f"astkat {command}"] + sys.argv[2:]
    module = importlib.import_module(f"astkat.{module_name}")
    result = getattr(module, function_name)()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
