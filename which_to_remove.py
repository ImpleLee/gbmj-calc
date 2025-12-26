from gbmj import parse_hand, Hand, ALL_牌, get_futures, branch_p, 牌
import gbmj
from tap import Tap
from collections import defaultdict
from rich import print
from typing import Literal

class Args(Tap):
  hand: Hand # Current hand
  seen: Hand = Hand() # Seen tiles
  why: Literal['tree', 'leaf'] | None = None # Explain why

  def configure(self):
    self.add_argument('hand', type=parse_hand)
    self.add_argument('--seen', type=parse_hand)

def main(args: Args):
  tree = get_futures(args.hand).prune()
  remain: Hand = ALL_牌 - args.hand - args.seen # type: ignore
  estimate: defaultdict[int, set[牌]] = defaultdict(set)
  assert isinstance(tree.next, dict)
  for tile, branch in tree.next.items():
    estimate[branch_p(branch, remain)].add(tile)
  values = sorted(estimate, reverse=True)
  for value in values[:5]:
    for tile in estimate[value]:
      print(f'{value}: {Hand([tile])}', end='')
      match args.why:
        case None:
          print()
        case 'tree':
          print()
          print(tree.next[tile])
        case 'leaf':
          print(' - ' + ' '.join(sorted(str(gbmj.__dict__[s[0]](*s[1:])[0]) for s in set().union(*(f.collect_leafs() for f in tree.next[tile].values())))))

if __name__ == "__main__":
  main(Args().parse_args())
