def replace_weights(self, gen_h5, disc_h5):
    """
    Replace generator and discriminator weights with
    weights stored in .h5 files.

    Args:
        gen_h5 (str): path to generator weights file
        disc_h5 (str): path to discriminator weights file
    """
    self.generator.load_weights(gen_h5)
    self.discriminator.load_weights(disc_h5)
