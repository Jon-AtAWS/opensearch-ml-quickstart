# Copyright opensearch-ml-quickstart contributors
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import json
import faker
import random
import logging


class QAndAFileReader:
    AMAZON_PQA_FILENAME_MAP = {
        "AMAZON_PQA_STREAMING_MEDIA_PLAYERS": "amazon_pqa_streaming_media_players.json",
        "AMAZON_PQA_HEADLIGHT_BULBS": "amazon_pqa_headlight_bulbs.json",
        "AMAZON_PQA_T_SHIRTS": "amazon_pqa_t-shirts.json",
        "AMAZON_PQA_MONITORS": "amazon_pqa_monitors.json",
        "AMAZON_PQA_IN_DASH_NAVIGATION": "amazon_pqa_in-dash_navigation.json",
        "AMAZON_PQA_ROUTERS": "amazon_pqa_routers.json",
        "AMAZON_PQA_TV_ANTENNAS": "amazon_pqa_tv_antennas.json",
        "AMAZON_PQA_SPORTS_WATER_BOTTLES": "amazon_pqa_sports_water_bottles.json",
        "AMAZON_PQA_REMOTE_CONTROLS": "amazon_pqa_remote_controls.json",
        "AMAZON_PQA_HEADLIGHT_AND_TAIL_LIGHT_CONVERSION_KITS": "amazon_pqa_headlight_&_tail_light_conversion_kits.json",
        "AMAZON_PQA_HOME_SECURITY_SYSTEMS": "amazon_pqa_home_security_systems.json",
        "AMAZON_PQA_GAMES": "amazon_pqa_games.json",
        "AMAZON_PQA_LED_AND_LCD_TVS": "amazon_pqa_led_&_lcd_tvs.json",
        "AMAZON_PQA_SMARTWATCHES": "amazon_pqa_smartwatches.json",
        "AMAZON_PQA_BATTERIES": "amazon_pqa_batteries.json",
        "AMAZON_PQA_JEANS": "amazon_pqa_jeans.json",
        "AMAZON_PQA_DOLLS": "amazon_pqa_dolls.json",
        "AMAZON_PQA_VEHICLE_BACKUP_CAMERAS": "amazon_pqa_vehicle_backup_cameras.json",
        "AMAZON_PQA_CAMCORDERS": "amazon_pqa_camcorders.json",
        "AMAZON_PQA_CHAIRS": "amazon_pqa_chairs.json",
        "AMAZON_PQA_QUADCOPTERS_AND_MULTIROTORS": "amazon_pqa_quadcopters_&_multirotors.json",
        "AMAZON_PQA_STRING_LIGHTS": "amazon_pqa_string_lights.json",
        "AMAZON_PQA_STANDS": "amazon_pqa_stands.json",
        "AMAZON_PQA_PORTABLE_BLUETOOTH_SPEAKERS": "amazon_pqa_portable_bluetooth_speakers.json",
        "AMAZON_PQA_SURVEILLANCE_DVR_KITS": "amazon_pqa_surveillance_dvr_kits.json",
        "AMAZON_PQA_ACCESSORIES": "amazon_pqa_accessories.json",
        "AMAZON_PQA_VIDEO_PROJECTORS": "amazon_pqa_video_projectors.json",
        "AMAZON_PQA_SUNGLASSES": "amazon_pqa_sunglasses.json",
        "AMAZON_PQA_FLOOR_MATS": "amazon_pqa_floor_mats.json",
        "AMAZON_PQA_ACTIVITY_AND_FITNESS_TRACKERS": "amazon_pqa_activity_&_fitness_trackers.json",
        "AMAZON_PQA_DIFFUSERS": "amazon_pqa_diffusers.json",
        "AMAZON_PQA_BULLET_CAMERAS": "amazon_pqa_bullet_cameras.json",
        "AMAZON_PQA_INKJET_PRINTERS": "amazon_pqa_inkjet_printers.json",
        "AMAZON_PQA_RECEIVERS": "amazon_pqa_receivers.json",
        "AMAZON_PQA_AREA_RUGS": "amazon_pqa_area_rugs.json",
        "AMAZON_PQA_OVER_EAR_HEADPHONES": "amazon_pqa_over-ear_headphones.json",
        "AMAZON_PQA_MATTRESSES": "amazon_pqa_mattresses.json",
        "AMAZON_PQA_HEADSETS": "amazon_pqa_headsets.json",
        "AMAZON_PQA_TRADITIONAL_LAPTOPS": "amazon_pqa_traditional_laptops.json",
        "AMAZON_PQA_DOME_CAMERAS": "amazon_pqa_dome_cameras.json",
        "AMAZON_PQA_PUMPS": "amazon_pqa_pumps.json",
        "AMAZON_PQA_SOUND_BARS": "amazon_pqa_sound_bars.json",
        "AMAZON_PQA_SPORTS_AND_ACTION_VIDEO_CAMERAS": "amazon_pqa_sports_&_action_video_cameras.json",
        "AMAZON_PQA_LED_STRIP_LIGHTS": "amazon_pqa_led_strip_lights.json",
        "AMAZON_PQA_PANELS": "amazon_pqa_panels.json",
        "AMAZON_PQA_TABLETS": "amazon_pqa_tablets.json",
        "AMAZON_PQA_MEMORY": "amazon_pqa_memory.json",
        "AMAZON_PQA_COSTUMES": "amazon_pqa_costumes.json",
        "AMAZON_PQA_SHEET_AND_PILLOWCASE_SETS": "amazon_pqa_sheet_&_pillowcase_sets.json",
        "AMAZON_PQA_WRIST_WATCHES": "amazon_pqa_wrist_watches.json",
        "AMAZON_PQA_BATTERY_CHARGERS": "amazon_pqa_battery_chargers.json",
        "AMAZON_PQA_ADAPTERS": "amazon_pqa_adapters.json",
        "AMAZON_PQA_TV_CEILING_AND_WALL_MOUNTS": "amazon_pqa_tv_ceiling_&_wall_mounts.json",
        "AMAZON_PQA_EXTERNAL_HARD_DRIVES": "amazon_pqa_external_hard_drives.json",
        "AMAZON_PQA_CRADLES": "amazon_pqa_cradles.json",
        "AMAZON_PQA_ON_DASH_CAMERAS": "amazon_pqa_on-dash_cameras.json",
        "AMAZON_PQA_LED_BULBS": "amazon_pqa_led_bulbs.json",
        "AMAZON_PQA_GUN_HOLSTERS": "amazon_pqa_gun_holsters.json",
        "AMAZON_PQA_TOWERS": "amazon_pqa_towers.json",
        "AMAZON_PQA_CODE_READERS_AND_SCAN_TOOLS": "amazon_pqa_code_readers_&_scan_tools.json",
        "AMAZON_PQA_KEYBOARDS": "amazon_pqa_keyboards.json",
        "AMAZON_PQA_GRAPHICS_CARDS": "amazon_pqa_graphics_cards.json",
        "AMAZON_PQA_EARBUD_HEADPHONES": "amazon_pqa_earbud_headphones.json",
        "AMAZON_PQA_USB_CABLES": "amazon_pqa_usb_cables.json",
        "AMAZON_PQA_POWER_CONVERTERS": "amazon_pqa_power_converters.json",
        "AMAZON_PQA_HEADLIGHT_ASSEMBLIES": "amazon_pqa_headlight_assemblies.json",
        "AMAZON_PQA_HANDHELD_FLASHLIGHTS": "amazon_pqa_handheld_flashlights.json",
        "AMAZON_PQA_MP3_AND_MP4_PLAYERS": "amazon_pqa_mp3_&_mp4_players.json",
        "AMAZON_PQA_CASES": "amazon_pqa_cases.json",
        "AMAZON_PQA_POSTERS_AND_PRINTS": "amazon_pqa_posters_&_prints.json",
        "AMAZON_PQA_LANDLINE_PHONES": "amazon_pqa_landline_phones.json",
        "AMAZON_PQA_SCREEN_PROTECTORS": "amazon_pqa_screen_protectors.json",
        "AMAZON_PQA_CHARGERS_AND_ADAPTERS": "amazon_pqa_chargers_&_adapters.json",
        "AMAZON_PQA_HOME_OFFICE_DESKS": "amazon_pqa_home_office_desks.json",
        "AMAZON_PQA_SETS": "amazon_pqa_sets.json",
        "AMAZON_PQA_SLR_CAMERA_LENSES": "amazon_pqa_slr_camera_lenses.json",
        "AMAZON_PQA_LIGHT_BARS": "amazon_pqa_light_bars.json",
        "AMAZON_PQA_WIGS": "amazon_pqa_wigs.json",
        "AMAZON_PQA_MASKS": "amazon_pqa_masks.json",
        "AMAZON_PQA_HAIR_EXTENSIONS": "amazon_pqa_hair_extensions.json",
        "AMAZON_PQA_BED_FRAMES": "amazon_pqa_bed_frames.json",
        "AMAZON_PQA_CAR": "amazon_pqa_car.json",
        "AMAZON_PQA_HIDDEN_CAMERAS": "amazon_pqa_hidden_cameras.json",
        "AMAZON_PQA_MOTHERBOARDS": "amazon_pqa_motherboards.json",
        "AMAZON_PQA_CONSOLES": "amazon_pqa_consoles.json",
        "AMAZON_PQA_UNLOCKED_CELL_PHONES": "amazon_pqa_unlocked_cell_phones.json",
        "AMAZON_PQA_AMAZON_ECHO_AND_ALEXA_DEVICES": "amazon_pqa_amazon_echo_&_alexa_devices.json",
        "AMAZON_PQA_FOOD_STORAGE_AND_ORGANIZATION_SETS": "amazon_pqa_food_storage_&_organization_sets.json",
        "AMAZON_PQA_REPELLENTS": "amazon_pqa_repellents.json",
        "AMAZON_PQA_USB_FLASH_DRIVES": "amazon_pqa_usb_flash_drives.json",
        "AMAZON_PQA_CAR_STEREO_RECEIVERS": "amazon_pqa_car_stereo_receivers.json",
        "AMAZON_PQA_CASUAL": "amazon_pqa_casual.json",
        "AMAZON_PQA_BASIC_CASES": "amazon_pqa_basic_cases.json",
        "AMAZON_PQA_COMPUTER_CASES": "amazon_pqa_computer_cases.json",
        "AMAZON_PQA_BACKPACKS": "amazon_pqa_backpacks.json",
        "AMAZON_PQA_FASHION_SNEAKERS": "amazon_pqa_fashion_sneakers.json",
        "AMAZON_PQA_PANTS": "amazon_pqa_pants.json",
        "AMAZON_PQA_BEDS": "amazon_pqa_beds.json",
        "AMAZON_PQA_CARRIER_CELL_PHONES": "amazon_pqa_carrier_cell_phones.json",
        "AMAZON_PQA_IN_DASH_DVD_AND_VIDEO_RECEIVERS": "amazon_pqa_in-dash_dvd_&_video_receivers.json",
    }

    AMAZON_PQA_CATEGORY_MAP = {
        "streaming media players": "AMAZON_PQA_STREAMING_MEDIA_PLAYERS",
        "headlight bulbs": "AMAZON_PQA_HEADLIGHT_BULBS",
        "t-shirts": "AMAZON_PQA_T_SHIRTS",
        "monitors": "AMAZON_PQA_MONITORS",
        "in-dash navigation": "AMAZON_PQA_IN_DASH_NAVIGATION",
        "routers": "AMAZON_PQA_ROUTERS",
        "tv antennas": "AMAZON_PQA_TV_ANTENNAS",
        "sports water bottles": "AMAZON_PQA_SPORTS_WATER_BOTTLES",
        "remote controls": "AMAZON_PQA_REMOTE_CONTROLS",
        "headlight and tail light conversion kits": "AMAZON_PQA_HEADLIGHT_AND_TAIL_LIGHT_CONVERSION_KITS",
        "home security systems": "AMAZON_PQA_HOME_SECURITY_SYSTEMS",
        "games": "AMAZON_PQA_GAMES",
        "led and lcd tvs": "AMAZON_PQA_LED_AND_LCD_TVS",
        "smartwatches": "AMAZON_PQA_SMARTWATCHES",
        "batteries": "AMAZON_PQA_BATTERIES",
        "jeans": "AMAZON_PQA_JEANS",
        "dolls": "AMAZON_PQA_DOLLS",
        "vehicle backup cameras": "AMAZON_PQA_VEHICLE_BACKUP_CAMERAS",
        "camcorders": "AMAZON_PQA_CAMCORDERS",
        "chairs": "AMAZON_PQA_CHAIRS",
        "quadcopters and multirotors": "AMAZON_PQA_QUADCOPTERS_AND_MULTIROTORS",
        "string lights": "AMAZON_PQA_STRING_LIGHTS",
        "stands": "AMAZON_PQA_STANDS",
        "portable bluetooth speakers": "AMAZON_PQA_PORTABLE_BLUETOOTH_SPEAKERS",
        "surveillance dvr kits": "AMAZON_PQA_SURVEILLANCE_DVR_KITS",
        "accessories": "AMAZON_PQA_ACCESSORIES",
        "video projectors": "AMAZON_PQA_VIDEO_PROJECTORS",
        "sunglasses": "AMAZON_PQA_SUNGLASSES",
        "floor mats": "AMAZON_PQA_FLOOR_MATS",
        "activity and fitness trackers": "AMAZON_PQA_ACTIVITY_AND_FITNESS_TRACKERS",
        "diffusers": "AMAZON_PQA_DIFFUSERS",
        "bullet cameras": "AMAZON_PQA_BULLET_CAMERAS",
        "inkjet printers": "AMAZON_PQA_INKJET_PRINTERS",
        "receivers": "AMAZON_PQA_RECEIVERS",
        "area rugs": "AMAZON_PQA_AREA_RUGS",
        "over-ear headphones": "AMAZON_PQA_OVER_EAR_HEADPHONES",
        "mattresses": "AMAZON_PQA_MATTRESSES",
        "headsets": "AMAZON_PQA_HEADSETS",
        "traditional laptops": "AMAZON_PQA_TRADITIONAL_LAPTOPS",
        "dome cameras": "AMAZON_PQA_DOME_CAMERAS",
        "pumps": "AMAZON_PQA_PUMPS",
        "sound bars": "AMAZON_PQA_SOUND_BARS",
        "sports and action video cameras": "AMAZON_PQA_SPORTS_AND_ACTION_VIDEO_CAMERAS",
        "led strip lights": "AMAZON_PQA_LED_STRIP_LIGHTS",
        "panels": "AMAZON_PQA_PANELS",
        "tablets": "AMAZON_PQA_TABLETS",
        "memory": "AMAZON_PQA_MEMORY",
        "costumes": "AMAZON_PQA_COSTUMES",
        "sheet and pillowcase sets": "AMAZON_PQA_SHEET_AND_PILLOWCASE_SETS",
        "wrist watches": "AMAZON_PQA_WRIST_WATCHES",
        "battery chargers": "AMAZON_PQA_BATTERY_CHARGERS",
        "adapters": "AMAZON_PQA_ADAPTERS",
        "tv ceiling and wall mounts": "AMAZON_PQA_TV_CEILING_AND_WALL_MOUNTS",
        "external hard drives": "AMAZON_PQA_EXTERNAL_HARD_DRIVES",
        "cradles": "AMAZON_PQA_CRADLES",
        "on-dash cameras": "AMAZON_PQA_ON_DASH_CAMERAS",
        "led bulbs": "AMAZON_PQA_LED_BULBS",
        "gun holsters": "AMAZON_PQA_GUN_HOLSTERS",
        "towers": "AMAZON_PQA_TOWERS",
        "code readers and scan tools": "AMAZON_PQA_CODE_READERS_AND_SCAN_TOOLS",
        "keyboards": "AMAZON_PQA_KEYBOARDS",
        "graphics cards": "AMAZON_PQA_GRAPHICS_CARDS",
        "earbud headphones": "AMAZON_PQA_EARBUD_HEADPHONES",
        "usb cables": "AMAZON_PQA_USB_CABLES",
        "power converters": "AMAZON_PQA_POWER_CONVERTERS",
        "headlight assemblies": "AMAZON_PQA_HEADLIGHT_ASSEMBLIES",
        "handheld flashlights": "AMAZON_PQA_HANDHELD_FLASHLIGHTS",
        "mp3 and mp4 players": "AMAZON_PQA_MP3_AND_MP4_PLAYERS",
        "cases": "AMAZON_PQA_CASES",
        "posters and prints": "AMAZON_PQA_POSTERS_AND_PRINTS",
        "landline phones": "AMAZON_PQA_LANDLINE_PHONES",
        "screen protectors": "AMAZON_PQA_SCREEN_PROTECTORS",
        "chargers and adapters": "AMAZON_PQA_CHARGERS_AND_ADAPTERS",
        "home office desks": "AMAZON_PQA_HOME_OFFICE_DESKS",
        "sets": "AMAZON_PQA_SETS",
        "slr camera lenses": "AMAZON_PQA_SLR_CAMERA_LENSES",
        "light bars": "AMAZON_PQA_LIGHT_BARS",
        "wigs": "AMAZON_PQA_WIGS",
        "masks": "AMAZON_PQA_MASKS",
        "hair extensions": "AMAZON_PQA_HAIR_EXTENSIONS",
        "bed frames": "AMAZON_PQA_BED_FRAMES",
        "car": "AMAZON_PQA_CAR",
        "hidden cameras": "AMAZON_PQA_HIDDEN_CAMERAS",
        "motherboards": "AMAZON_PQA_MOTHERBOARDS",
        "consoles": "AMAZON_PQA_CONSOLES",
        "unlocked cell phones": "AMAZON_PQA_UNLOCKED_CELL_PHONES",
        "amazon echo and alexa devices": "AMAZON_PQA_AMAZON_ECHO_AND_ALEXA_DEVICES",
        "food storage and organization sets": "AMAZON_PQA_FOOD_STORAGE_AND_ORGANIZATION_SETS",
        "repellents": "AMAZON_PQA_REPELLENTS",
        "usb flash drives": "AMAZON_PQA_USB_FLASH_DRIVES",
        "car stereo receivers": "AMAZON_PQA_CAR_STEREO_RECEIVERS",
        "casual": "AMAZON_PQA_CASUAL",
        "basic cases": "AMAZON_PQA_BASIC_CASES",
        "computer cases": "AMAZON_PQA_COMPUTER_CASES",
        "backpacks": "AMAZON_PQA_BACKPACKS",
        "fashion sneakers": "AMAZON_PQA_FASHION_SNEAKERS",
        "pants": "AMAZON_PQA_PANTS",
        "beds": "AMAZON_PQA_BEDS",
        "carrier cell phones": "AMAZON_PQA_CARRIER_CELL_PHONES",
        "in-dash dvd and video receivers": "AMAZON_PQA_IN_DASH_DVD_AND_VIDEO_RECEIVERS",
    }

    def __init__(self, directory, max_number_of_docs=-1):
        self.asins = set()
        self.questions = set()
        self.amazon_pqa_constants = self.AMAZON_PQA_FILENAME_MAP.keys()
        self.directory = directory
        self.fake = faker.Faker()
        self.max_number_of_docs = max_number_of_docs

    def printable_category_names(self):
        return ', '.join(
            map(
                lambda x: f"'{x}'",
                self.AMAZON_PQA_CATEGORY_MAP.keys()
            )
        )

    def amazon_pqa_category_name_to_constant(self, category_name):
        category_constant = self.AMAZON_PQA_CATEGORY_MAP.get(category_name, None)
        if category_constant:
            return category_constant
        raise ValueError(f"Unknown category name {category_name}\nValid names: {self.printable_category_names()}")

    def amazon_pqa_constant_to_filename(self, constant_name):
        return self.AMAZON_PQA_FILENAME_MAP[constant_name]

    def amazon_pqa_constant_to_category_name(self, constant):
        for category_name, constant_name in self.AMAZON_PQA_CATEGORY_MAP.items():
            if constant_name == constant:
                return category_name
        return None

    def amazon_pqa_category_names(self):
        for category in self.AMAZON_PQA_CATEGORY_MAP.keys():
            yield category

    def file_size(self, category_name):
        filename = self.amazon_pqa_category_name_to_constant(category_name)
        if not filename:
            raise Exception(f"Unknown category name {category_name}")
        filename = self.amazon_pqa_constant_to_filename(filename)
        if not filename:
            raise Exception(f"Unknown filename for {category_name}")
        filename = f"{self.directory}/{filename}"
        logging.info(filename)
        return os.path.getsize(filename)

    # These 2 functions standardize naming between constants and file names.
    def _filename_to_constant_name(self, filename):
        filename = filename.upper()
        filename = filename.replace("./", "")
        filename = filename.replace(".JSON", "")
        filename = filename.replace("&", "AND")
        return filename.replace("-", "_")

    def _filename_to_constant_value(self, filename):
        filename = filename.replace("./", "")
        filename = filename.replace(".json", "")
        filename = filename.replace("&", "and")
        filename = filename.replace("amazon_pqa_", "")
        return filename.replace("_", " ")

    # These 2 functions are code-generating functions. The constants above for
    # AMAZON_PQA_CATEGORY_MAP and AMAZON_PQA_FILENAME_MAP are generated from
    # these functions. Keeping them for debug.
    def _map_categories_to_constants(self):
        print("  AMAZON_PQA_CATEGORY_MAP = {")
        filenames = glob.glob(os.path.join(self.directory, "*.json"))
        for filename in filenames:
            constant_name = self._filename_to_constant_name(filename)
            constant_value = self._filename_to_constant_value(filename)
            print(f"    '{constant_value}': '{constant_name}',")
        print("  }")

    def _map_constants_to_filenames(self):
        print("  AMAZON_PQA_FILENAME_MAP = {")
        filenames = glob.glob(os.path.join(self.directory, "*.json"))
        for filename in filenames:
            constant_name = self._filename_to_constant_name(filename)
            print(f"    '{constant_name}': '{filename[2:]}',")
        print("  }")

    def random_gender(self):
        val = random.random()
        if val > 0.95:
            return "other"
        if val > 0.475:
            return "female"
        return "male"

    def enrich_question(self, pqa_constant, question):
        question["category_name"] = self.amazon_pqa_constant_to_category_name(
            pqa_constant
        )

        # According to https://stackoverflow.com/questions/1316887/what-is-the-most-efficient-string-concatenation-method-in-python
        # the most efficient method is f strings, not + or +=
        question["bullets"] = f'{question["bullet_point1"]} {question["bullet_point2"]}'
        question[
            "bullets"
        ] = f'{question["bullets"]} {question["bullet_point3"]} {question["bullet_point4"]}'
        question["bullets"] = f'{question["bullets"]} {question["bullet_point5"]}'
        for bullet in [
            "bullet_point1",
            "bullet_point2",
            "bullet_point3",
            "bullet_point4",
            "bullet_point5",
        ]:
            question.pop(bullet)

        enriched_answers = list()
        for answer in question["answers"]:
            answer["name"] = self.fake.name()
            answer["user_lat"] = float(self.fake.latitude())
            answer["user_lon"] = float(self.fake.longitude())
            answer["gender"] = self.random_gender()
            answer["age"] = round(random.random() * 100)
            answer["product_rating"] = int((random.random() * 100) / 20) + 1
            enriched_answers.append(answer)
        question["answers"] = enriched_answers

        return question

    def questions_for_category(self, pqa_constant, enriched=True):
        number_of_docs = 0
        filename = self.AMAZON_PQA_FILENAME_MAP[pqa_constant]
        with open(self.directory + "/" + filename) as f:
            logging.info(f"Processing {filename}")
            for line in f:
                if enriched:
                    yield self.enrich_question(pqa_constant, json.loads(line))
                else:
                    yield json.loads(line)
                number_of_docs += 1
                if (self.max_number_of_docs > 0) and (
                    number_of_docs >= self.max_number_of_docs
                ):
                    break

    def questions_for_all_categories(self, enriched=True):
        for pqa_constant in self.amazon_pqa_constants:
            yield from self.questions_for_category(pqa_constant, enriched)
        return None

    def print_categories(self):
        for cat in self.AMAZON_PQA_CATEGORY_MAP.keys():
            print(cat)

    def is_category_name(self, category_name):
        return category_name in self.AMAZON_PQA_CATEGORY_MAP.keys()
