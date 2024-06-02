import torch


def feature_loss(fmap_r, fmap_g):
    """
    Calculate the feature loss between real and generated feature maps.

    Args:
        fmap_r (list): List of real feature maps.
        fmap_g (list): List of generated feature maps.

    Returns:
        torch.Tensor: Feature loss.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Calculate the discriminator loss.

    Args:
        disc_real_outputs (list): List of real discriminator outputs.
        disc_generated_outputs (list): List of generated discriminator outputs.

    Returns:
        tuple: Tuple containing the total discriminator loss, real discriminator losses, and generated discriminator losses.
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Calculate the generator loss.

    Args:
        disc_outputs (list): List of discriminator outputs.

    Returns:
        tuple: Tuple containing the total generator loss and individual generator losses.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
