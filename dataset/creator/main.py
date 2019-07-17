from argparse import ArgumentParser
import from_images

if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    func_dict = dict()

    from_images.add_command_line(subparsers)
    func_dict.update(**from_images.func_dict)
    args = parser.parse_args()
    func = func_dict[args.command]
    del args.command
    func(**vars(args))
