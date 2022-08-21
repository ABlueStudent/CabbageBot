import re
import requests
import tensorflow as tf
import discord
import cv2
import numpy as np

token = str()
GUILD = list()
model = tf.keras.models.load_model("model")
tags = ["medianly damaged", "slightly damaged", "undamaged", "very damaged"]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:103.0) Gecko/20100101 Firefox/103.0"
}

with open("config", mode="r", encoding="utf-8") as file:
    for line in file.readlines():
        formatted = line.strip().split("=", maxsplit=1)
        if formatted[0] == "token":
            token = formatted[1]
        elif formatted == "GUILD":
            GUILD.append(discord.Object(id=formatted[1]))


class CustomClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message: discord.Message):
        content = [message.content]+[x.url for x in message.attachments]
        print("[{}:{}]{}:{} {}".format(
            message.guild.name,
            message.channel.name,
            message.author.display_name,
            ";".join(content),
            message.created_at
        ))

        # we do not want the bot to reply to itself
        if message.author.id == self.user.id:
            return

        if message.channel.name == "cabbage-bot" and len(message.attachments) != 0:
            predicts = list()
            for attachment in message.attachments:
                url: str = attachment.url
                if len(re.findall(r"[\w-]+\.(jpg|jpeg|png)$", url, re.IGNORECASE)) == 0:
                    continue

                ans = await self.predict(
                    await self.preproc(
                        url
                    )
                )

                if ans[1] < 0.5:
                    predicts.append("uncertain")
                else:
                    predicts.append(tags[ans[0]])

            if len(predicts) != 0:
                await message.reply(",".join(predicts), mention_author=True)

    async def preproc(self, url: str):
        try:
            img = cv2.imdecode(
                np.asarray(
                    bytearray(
                        requests.get(url, headers=headers).content
                    )
                ),
                -1
            )

            return cv2.cvtColor(
                cv2.resize(
                    img,
                    (512, 512),
                    interpolation=cv2.INTER_AREA
                ),
                cv2.COLOR_BGR2RGB
            )
        except cv2.error:
            print("failed to preproc url")

    async def predict(self, img: cv2.Mat):
        tensor = np.expand_dims(
            np.asarray(img),
            axis=0
        )
        result = enumerate(
            model.predict(tensor)[0]
        )
        return max(result, key=lambda x: x[1])


client = CustomClient()
client.run(token)
