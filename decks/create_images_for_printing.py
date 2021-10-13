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

from PIL import Image
from PIL.ExifTags import TAGS
import os
from pathlib import Path
import shutil
import cv2
import argparse
import numpy as np
from slugify import slugify
from sqlalchemy import create_engine
from tqdm import tqdm
import pandas as pd
import requests
import requests_cache
import json
import logging

logger = logging.getLogger(__name__)

engine = create_engine("postgresql+psycopg2://mtg:mtg@localhost:5432/mtg")

tqdm.pandas(desc="Progress")
requests_cache.install_cache('scryfall_cache', backend='sqlite', expire_after=24*60*60)

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
    Path("./card_images").mkdir(parents=True, exist_ok=True)
    path = f"./card_images/{card_name}.png"
    filename = os.path.realpath(path)

    # Download the file if it does not exist
    if os.path.isfile(filename):
        return path

    r = requests.get(
        f"https://api.scryfall.com/cards/{scryfall_id}",
        data=dict(format="image", version="png"),
        stream=True,
    )
    if r.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in r:
                f.write(chunk)

        return path

    raise Exception("Could not download os save card image")


def pad_with_borders(img, h, w, startX, endX, startY, endY, max_h=max_h, max_w=max_w):
    """Pad image with borders to achieve a given height and width
    :return img and new box points"""
    top = bottom = int((max_h - h) / 2)
    right = left = int((max_w - w) / 2)
    new_image = cv2.copyMakeBorder(
        img,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
    )
    new_startX = startX + left
    new_endX = endX + left
    new_startY = startY + top
    new_endY = endY + top

    return new_image, new_startX, new_endX, new_startY, new_endY


def crop_to_stadard(
    img,
    startX,
    endX,
    startY,
    endY,
    startX_std=startX_std,
    endX_std=endX_std,
    startY_std=startY_std,
    endY_std=endY_std,
):
    if startX >= startX_std:
        new_image = img[:, startX - startX_std :].copy()
        new_image = cv2.copyMakeBorder(
            new_image,
            top=0,
            bottom=0,
            left=0,
            right=startX - startX_std,
            borderType=cv2.BORDER_CONSTANT,
        )
    else:
        new_image = img[:, : startX - startX_std].copy()
        new_image = cv2.copyMakeBorder(
            new_image,
            top=0,
            bottom=0,
            left=startX_std - startX,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
        )

    if startY >= startY_std:
        new_image = new_image[startY - startY_std :, :].copy()
        new_image = cv2.copyMakeBorder(
            new_image,
            top=0,
            bottom=startY - startY_std,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
        )
    else:
        new_image = new_image[: startY - startY_std, :].copy()
        new_image = cv2.copyMakeBorder(
            new_image,
            top=startY_std - startY,
            bottom=0,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
        )

    return new_image


def enhance_with_scryfall_api(df: pd.DataFrame) -> pd.DataFrame:
    """Add current scryfall data about the card"""
    df["scryfall_data"] = df["scryfallId"].progress_apply(
        get_card_updated_data,
    )
    return df


def add_price_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Add current scryfall USD prices"""
    df["price_usd"] = df["scryfall_data"].progress_apply(
        lambda x: x.get("price", {}).get("usd", np.nan),
    )
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
    """Adds columns img_cv2, height and width"""
    df["img_cv2"] = df["image_local_path"].progress_apply(lambda x: cv2.imread(x))
    df["img_cv2"] = df["img_cv2"].fillna(
        df["image_local_path"].progress_apply(lambda x: cv2.imread(os.path.realpath(x)))
    )
    if pd.isnull(df["img_cv2"]).any():
        problems = df[pd.isnull(df["img_cv2"])]
        problems.to_csv('problems.csv')
        raise Exception('There are empty images in the dataframe. See "problems".')
    
    df["height"] = df["img_cv2"].progress_apply(
        lambda x: x.shape[0] if x is not None else np.nan
    )
    df["width"] = df["img_cv2"].progress_apply(
        lambda x: x.shape[1] if x is not None else np.nan
    )

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
    problems = df[pd.isnull(df["height"])]
    problems2 = df[pd.isnull(df["aspect_ratio_correct_by_proportion"])]
    df["new_height"] = df["height"].where(
        ~df["aspect_ratio_higher"],
        np.int((df["height"] * df["aspect_ratio_correct_by_proportion"])),
    )
    # shrink width if aspect ration is lower than expected
    df["new_edith"] = df["width"].where(
        df["aspect_ratio_higher"],
        np.int((df["width"] * df["aspect_ratio_correct_by_proportion"])),
    )

    return df


def generate_deck_img_dir(df: pd.DataFrame, deck_slug: str) -> pd.DataFrame:
    """Generates images in deck image dirs, risezed if needed"""
    Path(f"./deck_images/{deck_slug}").mkdir(parents=True, exist_ok=True)
    df["deck_image_path"] = df.apply(
        lambda row: f"./deck_images/{deck_slug}/{row.card_id_in_deck}-{row.card_name}-{'resized' if row.should_resize else ''}",
        axis="columns",
    )

    df["img_to_save"] = df.apply(
        lambda row: row.img_cv2
        if row.should_resize
        else cv2.resize(
            row.img_cv2, (row.new_height, row.new_width), interpolation=cv2.INTER_AREA
        ),
        axis="columns",
    )

    df.apply(
        lambda row: cv2.imwrite(row.deck_image_path, row.img_to_save),
        axis="columns",
    )

    return df


if __name__ == "__main__":
    pass

    DECK_SLUG = "00deck_passarinhos_as_is"

    deck_df_query = f"""
    SELECT deck_id, card_id_in_deck, card_name, cards."scryfallId"
    FROM decks
    JOIN cards
    ON decks.card_name=cards.name
    WHERE deck_id='{DECK_SLUG}'
    """
    deck_df = pd.read_sql_query(deck_df_query, engine)

    logger.info(f"Downloading images for deck {DECK_SLUG}")
    deck_df["image_local_path"] = deck_df.apply(
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
