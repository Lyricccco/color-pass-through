from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create/record observer calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    grid = subparsers.add_parser("grid", help="Print the 125 candidate phi values")
    grid.add_argument("--low", type=float, default=0.025)
    grid.add_argument("--high", type=float, default=0.075)
    grid.add_argument("--step", type=float, default=0.0125)
    record = subparsers.add_parser("record", help="Record a selected phi")
    record.add_argument("--observer-id", required=True)
    record.add_argument("--observer-type", choices=["human", "digital"], required=True)
    record.add_argument("--device-id", required=True)
    record.add_argument("--phi", nargs=3, type=float, required=True)
    record.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "grid":
        from color_pass_through.calibration import generate_phi_grid

        for index, phi in enumerate(
            generate_phi_grid(args.low, args.high, args.step), start=1
        ):
            print(f"{index:03d}: {phi[0]:.4f} {phi[1]:.4f} {phi[2]:.4f}")
        return
    from color_pass_through.calibration.manifest import save_calibration

    target = save_calibration(
        args.output,
        args.observer_id,
        args.device_id,
        args.phi,
        args.observer_type,
    )
    print(target)


if __name__ == "__main__":
    main()
