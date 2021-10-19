# -*- coding: utf-8 -*-

"""
This file has a function that, given a deck:
1. Pulls the images of all the cards in that deck (high res)
2. Resize the image to be 63mm x 88mm (same ration at least)
3. Pads the image with a dark grey border to serve as cut marker
4. Stores all the images in  folder named after the deck
5. Repeating cards get a number after it to identify them
6. The back of the card should also be saved somewhere
"""
from mtgnlp import config
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd
import requests
import requests_cache
import json
import logging

logPathFileName = config.LOGS_DIR.joinpath("decks_create_images_for_printing.log")

# create logger'
logger = logging.getLogger("decks_create_images_for_printing")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{logPathFileName}", mode="w")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

engine = create_engine(config.DB_STR)

tqdm.pandas(desc="Progress")
requests_cache.install_cache(
    "scryfall_cache", backend="sqlite", expire_after=24 * 60 * 60
)

# import the necessary packages

# defining argument parsers
# ap  = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,help="Input image path")
# args = vars(ap.parse_args())

startX_std, startY_std, endX_std, endY_std = 1295, 238, 1553, 615
width_std = endX_std - startX_std
height_std = endY_std - startY_std

max_h, max_w = 2560 + 1000, 3264


def get_card_updated_data(scryfall_id) -> str:
    """Gets updated data from scryfall

    Args:
        scryfall_id ([type]): [description]

    Returns:
        json: result from scryfalls card api

    ref: https://scryfall.com/docs/api/cards/id
    ex scryfall_id="e9d5aee0-5963-41db-a22b-cfea40a967a3"
    """
    r = requests.get(
        f"https://api.scryfall.com/cards/{scryfall_id}",
    )
    if r.status_code == 200:

        return json.loads(r.text)

    raise Exception("Could not download os save card image")


def download_card_image(scryfall_id, card_name: str) -> str:
    """Downloads card image from scryfall with best resolution
    and saves to ./card_images/{card_name}.png

    Args:
        scryfall_id ([type]): [description]
        card_name (str): [description]

    Returns:
        str: path to card relative to this scripts location
    """
    config.DECKS_DIR.joinpath("card_images").mkdir(parents=True, exist_ok=True)
    path = config.DECKS_DIR.joinpath(f"card_images/{card_name}.png")
    filename = os.path.realpath(path)

    # Download the file if it does not exist
    if os.path.isfile(filename):
        return path

    r = requests.get(
        f"https://api.scryfall.com/cards/{scryfall_id}?format=image&version=png",
        stream=True,
    )
    if r.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in r:
                f.write(chunk)

        return path

    raise Exception("Could not download os save card image")


def enhance_with_scryfall_api(df: pd.DataFrame) -> pd.DataFrame:
    """Add current scryfall data about the card"""
    df["scryfall_data"] = df["scryfallId"].progress_apply(
        get_card_updated_data,
    )
    return df


def add_price_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Add current scryfall USD prices"""
    df["price_usd"] = df["scryfall_data"].progress_apply(
        lambda x: x.get("prices", {}).get("usd", np.nan),
    )
    df["price_usd"] = pd.to_numeric(df["price_usd"])
    return df


def add_img_paths_col(df: pd.DataFrame) -> pd.DataFrame:
    """Downloads images and saves local path to
    image_local_path column
    """
    df["image_local_path"] = df.progress_apply(
        lambda row: download_card_image(row.scryfallId, card_name=row.card_name),
        axis="columns",
    )
    return df


def add_image_height_and_width(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns image, height and width"""
    df["image"] = df["image_local_path"].progress_apply(lambda x: Image.open(x))
    if pd.isnull(df["image"]).any():
        problems = df[pd.isnull(df["image"])]
        problems.to_csv("problems.csv")
        raise Exception('There are empty images in the dataframe. See "problems".')

    df["height"] = df["image"].progress_apply(lambda x: x.size[1])
    df["width"] = df["image"].progress_apply(lambda x: x.size[0])

    return df


def should_correct_aspect_ratio(
    df: pd.DataFrame, tolerance: float = 0.02
) -> pd.DataFrame:
    """If image file does not meet aspect ration 88/63,
    resize the image
    Tolerates by default 2% difference in aspect ratio

    ref: https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    """
    EXPECTED_RATIO = 88 / 63
    df["aspect_ratio"] = df["height"] / df["width"]
    df["aspect_ratio_higher"] = df["aspect_ratio"] > EXPECTED_RATIO
    df["aspect_ratio_correct_by_proportion"] = df["aspect_ratio"] / EXPECTED_RATIO
    df["should_resize"] = (
        np.abs(df["aspect_ratio"] - EXPECTED_RATIO) / (EXPECTED_RATIO) > tolerance
    )

    # shrink height if aspect ration is higher than expected
    df["new_height"] = df["height"].where(
        ~df["aspect_ratio_higher"],
        (df["height"] * df["aspect_ratio_correct_by_proportion"]).apply(int),
    )
    # shrink width if aspect ration is lower than expected
    df["new_width"] = df["width"].where(
        df["aspect_ratio_higher"],
        (df["width"] * df["aspect_ratio_correct_by_proportion"]).apply(int),
    )

    return df


def generate_deck_img_dir(df: pd.DataFrame, deck_slug: str) -> pd.DataFrame:
    """Generates images in deck image dirs, risezed if needed"""
    logger.info("generate_deck_img_dir")
    Path(f"./deck_images/{deck_slug}").mkdir(parents=True, exist_ok=True)
    df["deck_image_path"] = df.progress_apply(
        lambda row: f"./deck_images/{deck_slug}/{row.card_id_in_deck}-{row.card_name}-{'resized' if row.should_resize else ''}.png",
        axis="columns",
    )

    logger.info("Start resizing images")
    df["deck_image"] = df.progress_apply(
        lambda row: row.image
        if not row.should_resize
        else row.image.resize((row.new_width, row.new_height)),
        axis="columns",
    )

    logger.info("Saving resized images")
    df.progress_apply(
        lambda row: row.deck_image.save(row.deck_image_path)
        if not os.path.isfile(row.deck_image_path)
        else np.nan,
        axis="columns",
    )

    return df


if __name__ == "__main__":
    pass

    DECK_SLUG = "00deck_passarinhos_as_is"

    deck_df_query = f"""
    SELECT deck_id, card_id_in_deck, card_name, cards."scryfallId"
    FROM "{config.DECKS_TNAME}"
    JOIN "{config.CARDS_TNAME}"
    ON "{config.DECKS_TNAME}".card_name="{config.CARDS_TNAME}".name
    WHERE deck_id='{DECK_SLUG}'
    """
    deck_df = pd.read_sql_query(deck_df_query, engine)

    logger.info(f"Downloading images for deck {DECK_SLUG}")
    deck_df["image_local_path"] = deck_df.progress_apply(
        lambda row: download_card_image(row.scryfallId, card_name=row.card_name),
        axis="columns",
    )

    deck_df = (
        deck_df.pipe(enhance_with_scryfall_api)
        .pipe(add_price_usd)
        .pipe(add_img_paths_col)
        .pipe(add_image_height_and_width)
        .pipe(should_correct_aspect_ratio)
        .pipe(generate_deck_img_dir, deck_slug=DECK_SLUG)
        # .pipe(position_imgs_in_A4)
    )

    logger.debug(deck_df.price_usd)
    logger.info(f"Deck estimated price: {deck_df.price_usd.sum()} USD")

    a = 1

    # for filename in tqdm(os.listdir('./images')):
    #     if True in [x in filename for x in to_skip] or 'AVI' in filename:
    #         continue

    #     # procesess the import image with opencv
    #     # img_path = args.get("image")
    #     img_path = './images/' + filename
    #     image = cv2.imread(img_path)

    #     imagePIL = Image.open(img_path)
    #     exif = {
    #         TAGS[k]: v
    #         for k, v in imagePIL._getexif().items()
    #         if k in TAGS
    #     }
    #     imagePIL.close()

    #     if image is None:
    #         deal_with_exception(img_path)
    #         continue

    #     # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    #     (h, w) = image.shape[:2]
    #     if (h > max_h) or (w > max_w):
    #         raise Exception(f'Picture with h={h} or w={w} greater than max_h={max_h} or max_w={max_w}')

    #     cv2.imwrite(f'./2faces/{img_path.split("/")[-1]}', new_image)
