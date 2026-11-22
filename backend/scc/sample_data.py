"""Sample benchmark images and dataset presets for interactive testing and batch evaluation.
Generates distinct visual scenes with ground-truth labels that CLIP ViT-B/32 recognizes reliably.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFilter


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

ANIMALS_CLASSES = [
    "golden retriever dog",
    "gray timber wolf",
    "tabby domestic cat",
    "spotted deer in forest",
    "colorful tropical bird",
    "green tree frog",
    "brown galloping horse",
    "african lion in savanna",
]

VEHICLES_CLASSES = [
    "commercial passenger airplane",
    "modern passenger automobile",
    "large cargo freight truck",
    "ocean vessel freight ship",
    "supersonic military fighter jet",
    "two wheeled racing motorcycle",
    "city passenger transport bus",
    "yellow construction bulldozer",
]


def create_dog_image() -> Image.Image:
    """Creates a high-contrast warm golden retriever dog illustration with snout, floppy ears, eyes."""
    img = Image.new("RGB", (224, 224), (230, 240, 235))
    draw = ImageDraw.Draw(img)

    # Background meadow gradient
    for y in range(224):
        g = int(210 - y * 0.3)
        draw.line([(0, y), (223, y)], fill=(180, max(g, 120), 100))

    # Dog body / torso
    draw.ellipse([50, 100, 174, 220], fill=(210, 150, 80))
    # Chest highlight
    draw.ellipse([75, 120, 149, 210], fill=(240, 205, 140))

    # Dog head
    draw.ellipse([67, 45, 157, 135], fill=(215, 155, 85))

    # Floppy ears
    draw.polygon([(65, 55), (45, 110), (75, 105)], fill=(175, 110, 50))
    draw.polygon([(159, 55), (179, 110), (149, 105)], fill=(175, 110, 50))

    # Muzzle / snout
    draw.ellipse([87, 80, 137, 125], fill=(245, 215, 160))

    # Black nose
    draw.ellipse([102, 85, 122, 100], fill=(20, 15, 15))
    draw.line([(112, 100), (112, 112)], fill=(20, 15, 15), width=2)
    # Mouth line
    draw.arc([97, 102, 127, 118], 20, 160, fill=(20, 15, 15), width=2)

    # Eyes
    draw.ellipse([82, 65, 96, 77], fill=(45, 25, 15))
    draw.ellipse([86, 67, 90, 71], fill=(255, 255, 255))  # highlight
    draw.ellipse([128, 65, 142, 77], fill=(45, 25, 15))
    draw.ellipse([132, 67, 136, 71], fill=(255, 255, 255))  # highlight

    # Soft blur to blend
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_wolf_image() -> Image.Image:
    """Creates a wild gray timber wolf with pointed upright ears, amber eyes, and silver-gray coat."""
    img = Image.new("RGB", (224, 224), (200, 210, 225))
    draw = ImageDraw.Draw(img)

    # Snowy pine forest background
    for y in range(224):
        b = int(190 + y * 0.15)
        draw.line([(0, y), (223, y)], fill=(160, 180, min(b, 240)))

    # Pointed pine trees in background
    for x_tree in [25, 80, 195]:
        draw.polygon([(x_tree, 20), (x_tree - 25, 110), (x_tree + 25, 110)], fill=(70, 95, 85))

    # Wolf neck and shoulders
    draw.ellipse([45, 110, 179, 230], fill=(120, 125, 135))
    draw.ellipse([65, 130, 159, 225], fill=(185, 190, 195))

    # Wolf head (wedge shape)
    draw.ellipse([67, 50, 157, 140], fill=(130, 135, 145))

    # Pointed upright ears
    draw.polygon([(70, 65), (60, 20), (95, 55)], fill=(100, 105, 115))
    draw.polygon([(72, 62), (68, 30), (90, 56)], fill=(210, 190, 190))

    draw.polygon([(154, 65), (164, 20), (129, 55)], fill=(100, 105, 115))
    draw.polygon([(152, 62), (156, 30), (134, 56)], fill=(210, 190, 190))

    # Snout
    draw.polygon([(82, 85), (142, 85), (112, 130)], fill=(175, 178, 180))
    # Black nose
    draw.ellipse([103, 115, 121, 128], fill=(15, 15, 20))

    # Amber / golden eyes
    draw.ellipse([80, 72, 94, 82], fill=(215, 155, 30))
    draw.ellipse([85, 74, 89, 80], fill=(10, 10, 10))  # pupil
    draw.ellipse([130, 72, 144, 82], fill=(215, 155, 30))
    draw.ellipse([135, 74, 139, 80], fill=(10, 10, 10))  # pupil

    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def create_airplane_image() -> Image.Image:
    """Creates a commercial passenger jet airliner flying in blue sky with white clouds."""
    img = Image.new("RGB", (224, 224), (110, 175, 235))
    draw = ImageDraw.Draw(img)

    # Blue sky gradient
    for y in range(224):
        r = int(90 + y * 0.4)
        g = int(160 + y * 0.3)
        b = int(235)
        draw.line([(0, y), (223, y)], fill=(min(r, 220), min(g, 235), b))

    # Fluffy clouds
    for cx, cy in [(40, 170), (80, 160), (180, 180), (140, 190)]:
        draw.ellipse([cx - 40, cy - 25, cx + 40, cy + 25], fill=(245, 250, 255))

    # Airplane swept wings
    draw.polygon([(100, 105), (15, 150), (45, 155), (125, 115)], fill=(195, 200, 210))  # left wing
    draw.polygon([(120, 105), (205, 70), (195, 60), (105, 95)], fill=(215, 220, 228))   # right wing

    # Fuselage (airliner body)
    draw.ellipse([45, 95, 185, 125], fill=(250, 252, 255))

    # Cockpit windows
    draw.polygon([(170, 102), (183, 105), (178, 110), (167, 107)], fill=(40, 60, 90))

    # Passenger windows row
    for wx in range(80, 160, 10):
        draw.ellipse([wx, 106, wx + 4, 111], fill=(50, 80, 120))

    # Tail fin / vertical stabilizer
    draw.polygon([(50, 100), (30, 50), (55, 50), (75, 100)], fill=(210, 45, 45))  # red tail fin

    # Jet engines under wings
    draw.ellipse([65, 125, 95, 137], fill=(130, 135, 145))
    draw.ellipse([135, 90, 165, 102], fill=(130, 135, 145))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def create_automobile_image() -> Image.Image:
    """Creates a modern passenger automobile car on an asphalt road."""
    img = Image.new("RGB", (224, 224), (205, 225, 245))
    draw = ImageDraw.Draw(img)

    # Sky
    for y in range(130):
        draw.line([(0, y), (223, y)], fill=(170 + int(y * 0.4), 205 + int(y * 0.2), 240))

    # Road & mountains
    draw.rectangle([0, 130, 223, 223], fill=(65, 70, 75))
    draw.line([(0, 175), (223, 175)], fill=(235, 200, 50), width=3)  # yellow road stripe

    # Car body (red sports coupe)
    draw.ellipse([40, 115, 184, 175], fill=(215, 35, 40))
    # Roof cabin
    draw.polygon([(70, 125), (90, 85), (145, 85), (165, 125)], fill=(30, 35, 45))
    # Windshields
    draw.polygon([(76, 122), (92, 90), (115, 90), (115, 122)], fill=(170, 205, 235))
    draw.polygon([(119, 122), (119, 90), (142, 90), (158, 122)], fill=(170, 205, 235))

    # Wheels
    draw.ellipse([55, 145, 95, 185], fill=(25, 25, 30))
    draw.ellipse([65, 155, 85, 175], fill=(180, 185, 190))  # rim
    draw.ellipse([135, 145, 175, 185], fill=(25, 25, 30))
    draw.ellipse([145, 155, 165, 175], fill=(180, 185, 190))  # rim

    # Headlights & grille
    draw.ellipse([175, 130, 185, 142], fill=(255, 245, 180))
    draw.rectangle([165, 145, 182, 152], fill=(35, 35, 40))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def create_cat_image() -> Image.Image:
    """Creates a striped tabby cat with green eyes and whiskers."""
    img = Image.new("RGB", (224, 224), (245, 235, 220))
    draw = ImageDraw.Draw(img)

    # Cat body
    draw.ellipse([55, 115, 169, 225], fill=(175, 135, 95))
    # Tabby stripes on body
    for sy in [140, 165, 190]:
        draw.arc([65, sy, 159, sy + 30], 20, 160, fill=(110, 75, 45), width=3)

    # Cat head
    draw.ellipse([70, 60, 154, 140], fill=(185, 145, 105))

    # Pointed cat ears
    draw.polygon([(72, 75), (65, 25), (100, 60)], fill=(160, 115, 75))
    draw.polygon([(75, 70), (70, 35), (95, 60)], fill=(235, 170, 175))  # pink inner
    draw.polygon([(152, 75), (159, 25), (124, 60)], fill=(160, 115, 75))
    draw.polygon([(149, 70), (154, 35), (129, 60)], fill=(235, 170, 175))

    # Green almond eyes
    draw.ellipse([83, 85, 101, 100], fill=(120, 195, 70))
    draw.ellipse([91, 86, 93, 99], fill=(10, 15, 10))  # slit pupil
    draw.ellipse([123, 85, 141, 100], fill=(120, 195, 70))
    draw.ellipse([131, 86, 133, 99], fill=(10, 15, 10))

    # Pink nose and mouth
    draw.polygon([(107, 107), (117, 107), (112, 114)], fill=(235, 140, 150))
    draw.arc([102, 110, 112, 122], 30, 150, fill=(60, 40, 25), width=2)
    draw.arc([112, 110, 122, 122], 30, 150, fill=(60, 40, 25), width=2)

    # Whiskers
    draw.line([(60, 110), (100, 112)], fill=(240, 240, 245), width=1)
    draw.line([(58, 117), (98, 116)], fill=(240, 240, 245), width=1)
    draw.line([(126, 112), (166, 110)], fill=(240, 240, 245), width=1)
    draw.line([(128, 116), (168, 117)], fill=(240, 240, 245), width=1)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def create_ship_image() -> Image.Image:
    """Creates a large container vessel cargo ship sailing on deep blue sea."""
    img = Image.new("RGB", (224, 224), (180, 215, 240))
    draw = ImageDraw.Draw(img)

    # Sky
    for y in range(120):
        draw.line([(0, y), (223, y)], fill=(160 + int(y * 0.4), 205 + int(y * 0.2), 245))

    # Deep blue ocean
    for y in range(120, 224):
        draw.line([(0, y), (223, y)], fill=(20, 60 + int((y - 120) * 0.4), 140 + int((y - 120) * 0.5)))

    # Ship hull (long container vessel)
    draw.polygon([(30, 150), (195, 150), (175, 180), (45, 180)], fill=(185, 30, 35))  # red waterline hull
    draw.polygon([(25, 135), (205, 135), (195, 150), (30, 150)], fill=(30, 35, 45))   # black upper hull

    # Cargo containers stack
    colors = [(45, 110, 180), (210, 140, 35), (45, 155, 80), (190, 50, 45)]
    for row in range(3):
        for col in range(6):
            c_idx = (row * 3 + col) % len(colors)
            cx = 45 + col * 20
            cy = 115 - row * 10
            draw.rectangle([cx, cy, cx + 18, cy + 8], fill=colors[c_idx])

    # Bridge tower superstructure
    draw.rectangle([165, 95, 190, 135], fill=(245, 245, 250))
    draw.rectangle([170, 100, 185, 105], fill=(45, 75, 110))  # bridge windows
    draw.rectangle([178, 80, 184, 95], fill=(210, 35, 35))   # funnel

    # Bow wave / white foam
    draw.ellipse([190, 145, 215, 155], fill=(240, 250, 255))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


SAMPLE_GENERATORS = {
    "dog": ("dog", CIFAR10_CLASSES, create_dog_image),
    "wolf": ("wolf", ANIMALS_CLASSES, create_wolf_image),
    "airplane": ("airplane", CIFAR10_CLASSES, create_airplane_image),
    "automobile": ("automobile", CIFAR10_CLASSES, create_automobile_image),
    "cat": ("cat", CIFAR10_CLASSES, create_cat_image),
    "ship": ("ship", CIFAR10_CLASSES, create_ship_image),
}


def get_sample_catalogue() -> List[Dict[str, Any]]:
    """Returns metadata for preloaded benchmark samples."""
    return [
        {
            "id": "dog",
            "name": "Golden Retriever (Dog)",
            "ground_truth": "dog",
            "preset": "cifar10",
            "classes": CIFAR10_CLASSES,
            "description": "High-contrast canine sample ideal for testing semantic drift toward wolf or cat under PGD.",
        },
        {
            "id": "wolf",
            "name": "Timber Wolf (Wolf)",
            "ground_truth": "gray timber wolf",
            "preset": "animals",
            "classes": ANIMALS_CLASSES,
            "description": "Fine-grained animal benchmark sample testing hardest-competitor separation.",
        },
        {
            "id": "airplane",
            "name": "Commercial Airliner (Airplane)",
            "ground_truth": "airplane",
            "preset": "cifar10",
            "classes": CIFAR10_CLASSES,
            "description": "Object recognition benchmark from CIFAR-10 / ImageNet aerial class.",
        },
        {
            "id": "automobile",
            "name": "Coupe Sports Car (Automobile)",
            "ground_truth": "automobile",
            "preset": "cifar10",
            "classes": CIFAR10_CLASSES,
            "description": "Vehicle classification benchmark testing boundary between automobile and truck.",
        },
        {
            "id": "cat",
            "name": "Domestic Tabby (Cat)",
            "ground_truth": "cat",
            "preset": "cifar10",
            "classes": CIFAR10_CLASSES,
            "description": "Feline class benchmark susceptible to dog/frog adversarial confusion.",
        },
        {
            "id": "ship",
            "name": "Container Vessel (Ship)",
            "ground_truth": "ship",
            "preset": "cifar10",
            "classes": CIFAR10_CLASSES,
            "description": "Maritime transport class evaluated on cross-modal margin recovery.",
        },
    ]


def get_sample_image(sample_id: str) -> Image.Image:
    """Generates the PIL image for the requested sample ID."""
    if sample_id in SAMPLE_GENERATORS:
        return SAMPLE_GENERATORS[sample_id][2]()
    return create_dog_image()
