import torch

from sentinel.main import gradient_attack, overlay_patch, universal_perturbation


def test_fgsm_uses_one_sign_step_and_respects_pixel_and_epsilon_bounds():
    source = torch.full((1, 3, 8, 8), 0.5)
    attacked, count = gradient_attack(lambda x: x.sum(), source, "FGSM", 0.04, 0.01, 20, False)
    assert count == 1
    assert torch.allclose(attacked, source + 0.04)
    assert float((attacked - source).abs().max()) <= 0.040001
    assert float(attacked.min()) >= 0 and float(attacked.max()) <= 1


def test_pgd_projects_every_step_and_reaches_budget_for_linear_objective():
    source = torch.full((1, 3, 8, 8), 0.5)
    attacked, count = gradient_attack(lambda x: x.sum(), source, "PGD", 0.05, 0.02, 5, True, seed=5)
    assert count == 5
    assert float((attacked - source).abs().max()) <= 0.050001
    assert float(attacked.min()) >= 0 and float(attacked.max()) <= 1
    assert torch.allclose(attacked, source + 0.05)


def test_attack_computes_input_gradient_without_updating_model_parameters():
    model = torch.nn.Linear(4, 1, bias=False)
    before = model.weight.detach().clone()
    source = torch.full((1, 4), 0.5)
    attacked, _ = gradient_attack(lambda x: model(x).sum(), source, "FGSM", 0.02, 0.01, 1, False)
    assert torch.equal(model.weight.detach(), before)
    assert model.weight.grad is None
    assert not attacked.requires_grad


def test_universal_attack_uses_the_same_shared_delta_for_each_image():
    images = torch.full((3, 3, 12, 12), 0.5)
    attacked, delta = universal_perturbation(lambda x: x.sum(), images, 0.04, 0.02, 3, False)
    actual_delta = attacked - images
    assert delta.shape == (1, 3, 12, 12)
    assert torch.allclose(actual_delta[0], actual_delta[1])
    assert torch.allclose(actual_delta[1], actual_delta[2])
    assert float(actual_delta.abs().max()) <= 0.040001


def test_patch_overlay_is_localized_and_respects_requested_corner():
    images = torch.zeros((2, 3, 20, 20))
    patch = torch.ones((1, 3, 6, 6))
    overlaid, mask = overlay_patch(images, patch, 0, 0)
    assert torch.all(overlaid[:, :, :6, :6] == 1)
    assert torch.all(overlaid[:, :, 10:, 10:] == 0)
    assert abs(float(mask.mean()) - 36 / 400) < 1e-6
