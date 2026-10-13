import random
from fake_platform import FakePlatform
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Action:
    type: str
    post_id: Optional[int] = None
    content: Optional[str] = None

class Bot:
    def __init__(self, username: str, persona: dict, platform) -> None:
        self.username = username
        self.persona = persona
        self.platform = platform
        self.seen = set()
        self.history = list()
    
    def read(self) -> dict:
        return {
            "feed": self.platform.feed_latest(n=10),
            "notifications": self.platform.get_notifications(self.username),
        }
    
    def think(self, obs: dict) -> Optional[Action]:
        candidates = [p for p in obs["feed"] if p["id"] not in self.seen and p["author"] != self.username]
        kind = random.choices(["like", "reply", "post", "none"], weights=[5, 2, 1, 2])[0]
        if kind in ("like", "reply") and not candidates:
            kind = "post"
            
        if kind in ("like", "reply"):
            target = random.choice(candidates)
        content = random.choice(self.persona["templates"])
        
        if kind == "none":
            return None
        elif kind == "like":
            return Action("like", post_id=target["id"])
        elif kind == "reply":
            return Action("reply", post_id=target["id"], content=content)
        else:
            return Action("post", content=content)
        
    def act(self, action: Action) -> None:
        if action.type == "like":
            self.platform.like(self.username, action.post_id)
        elif action.type == "reply":
            self.platform.reply(self.username, action.post_id, action.content)
        elif action.type == "post":
            self.platform.post(self.username, action.content)
        else:
            raise ValueError(f"unknown action type: {action.type}")

    def remember(self, action: Action) -> None:
        if action.type in ("like", "reply"):
            self.seen.add(action.post_id)
        self.history.append((datetime.now().isoformat(), action))
        
    def step(self) -> Optional[Action]:
        action = self.think(self.read())
        if action is None:
            return None
        self.act(action)
        self.remember(action)
        return action


if __name__ == "__main__":
    random.seed(0)
    p = FakePlatform()
    bots = [
        Bot("amara_nurse", {"templates": [
            "Ward 4 was full again last night.",
            "We still remember what 2021 felt like on the wards.",
        ]}, p),
        Bot("joe_cafe", {"templates": [
            "Checking a pass beats closing the shop again.",
            "My cafe barely survived the last lockdown.",
        ]}, p),
        Bot("lena_teacher", {"templates": [
            "Thirty kids in one room, every single day.",
            "Half my class was out sick last winter.",
        ]}, p),
    ]
    p.post("npc_seed", "Anyone else worried about the dolphin flu news?")

    for rnd in range(1, 11):
        for bot in bots:
            action = bot.step()
            if action is None:
                print(f"[round {rnd}] {bot.username} idle")
            else:
                target = f" -> #{action.post_id}" if action.post_id else ""
                text = f": {action.content}" if action.content else ""
                print(f"[round {rnd}] {bot.username} {action.type}{target}{text}")

    for post in p.posts.values():
        assert post["author"] not in post["likes"]
        if post["parent_id"] is not None:
            assert post["author"] != p.get_post(post["parent_id"])["author"]
    for bot in bots:
        ids = [a.post_id for _, a in bot.history if a.post_id is not None]
        assert len(ids) == len(set(ids))
        assert len(bot.history) <= 10

    print()
    for post in p.feed_latest(n=50):
        print(post["id"], post["author"], post["parent_id"], len(post["likes"]), post["content"])
    print({bot.username: len(bot.history) for bot in bots})
