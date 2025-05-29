from molmo_wrapper import MolmoWrapper

molmo = MolmoWrapper(headless=False)
coords = molmo.point_to_object("teapot.jpg", prompt="teapot")
