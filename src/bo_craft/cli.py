"""Trusted local CLI; proposal execution can be admitted as a command job."""

import argparse
import sys

from .api import capabilities, propose, recommend
from .io import save
from .specs import Error, canonical, load, validate_request


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bo-craft")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    for name in ["validate", "propose", "recommend"]:
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--output")
        if name == "recommend":
            p.add_argument("--kind", choices=["empirical", "model", "posterior"], default="empirical")
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            value = capabilities()
        else:
            request = load(args.input)
            value = {"valid": True} if args.command == "validate" and validate_request(request) else None
            if args.command == "propose":
                value = propose(request)
            elif args.command == "recommend":
                value = recommend(request, kind=args.kind)
            if args.output:
                save(args.output, value)
        print(canonical(value).decode())
        return 0
    except (Error, ImportError, OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, Error):
            code = exc.code
        elif isinstance(exc, ImportError):
            code = "missing_dependency"
        elif isinstance(exc, OSError):
            code = "io_error"
        else:
            code = "numerical_failure"
        print(canonical({"error": {"code": code, "message": str(exc)[:2048]}}).decode(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
