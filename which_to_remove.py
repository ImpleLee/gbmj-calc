import gbmj
from prompt_toolkit import PromptSession
from rich import print
from collections import defaultdict

def main():
  session = PromptSession()
  while True:
    try:
      hand = gbmj.parse_hand(session.prompt('Hand> '))
    except (KeyboardInterrupt, EOFError):
      break
    tree = gbmj.get_futures(hand).prune()
    remain = gbmj.ALL_牌 - hand
    estimate = defaultdict(set)
    for tile, branch in tree.next.items():
      estimate[gbmj.branch_p(branch, remain)].add(tile)
    values = sorted(estimate, reverse=True)
    for value in values[:5]:
      print(f"{value}: {estimate[value]}")

if __name__ == "__main__":
  main()
