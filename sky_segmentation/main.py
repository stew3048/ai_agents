try:
    import torch
except Exception:
    torch = None


def main() -> None:
    if torch is not None:
        _ = torch.empty(0)
    print("Pipeline ready")


if __name__ == "__main__":
    main()
