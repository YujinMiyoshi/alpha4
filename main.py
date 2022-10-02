import datetime
import re

import deepl
import discord
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

deepl_token = "07a4eaaa-2f43-593e-852b-680151a3fb98:fx"
discord_token = "MTAyNTY3ODc2OTc2OTQzNTE0OA.GzIW44.LOm_C8QF3u7E5_u-q_3MaCycCY3CnOKtVfqCPM"
token = "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"


def get_translation(text: str) -> str:
    result = translator.translate_text(text, target_lang="EN-US")
    return result.text  # type: ignore


def get_usage() -> str:
    usage = translator.get_usage()
    msg = ""
    if usage.any_limit_reached:
        msg += "Translation limit reached."
    if usage.character.valid:
        msg += f"Character usage: {usage.character.count} of {usage.character.limit}"
    if usage.document.valid:
        msg += f"Document usage: {usage.document.count} of {usage.document.limit}"
    return msg


class StableDiffusion:
    def __init__(self, token) -> None:
        model_id = "CompVis/stable-diffusion-v1-4"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=token)

    def generate_image(self, text: str) -> str:
        pipe = self.pipe.to("cuda")
        with autocast("cuda"):
            image = pipe(text)["sample"][0]
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image.save(f"out/image_{now}.png")
        img_path = f"out/image_{now}.png"
        return img_path


if __name__ == "__main__":
    translator = deepl.Translator(deepl_token)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    sd = StableDiffusion(token)

    @client.event
    async def on_ready():
        print(f"We have logged in as {client.user}")

    @client.event
    async def on_message(message):
        if message.author.bot:
            return
        elif client.user in message.mentions:
            file = None
            t = re.sub(r"<@.*> ", "", message.content)
            if t == "残り":
                reply = get_usage()
            else:
                await message.channel.send(f"{message.author.mention} ちょっとまっててね。")
                print(f"Start generate image. {t}")
                text = get_translation(t)
                reply = f"{message.author.mention} {text}"
                file = discord.File(sd.generate_image(text))
            await message.channel.send(reply, file=file)

    client.run(discord_token)
