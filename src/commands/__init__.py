"""Command modules for train, embed, and eval."""

from src.commands.embed import cmd_embed
from src.commands.eval import cmd_eval
from src.commands.train import cmd_train

__all__ = ["cmd_embed", "cmd_eval", "cmd_train"]
