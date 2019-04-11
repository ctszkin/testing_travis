from src.pci import run_PCI, setup_pci, gen_figure

def test_pcia():
    X_train, Y_train, X_val, Y_val = setup_pci(seed=1, n=500)
    df = run_PCI(X_train, Y_train, X_val, Y_val)
    gen_figure(df)
