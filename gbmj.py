from collections import Counter, defaultdict
from typing import Final
from dataclasses import dataclass

牌 = tuple[int, str]
面子 = tuple[tuple[int, int, int], str]

class Hand(Counter[牌 | 面子, int]):
  def __repr__(self) -> str:
    ret = ''
    for (tiles, suit), count in self.items():
      if isinstance(tiles, int):
        continue
      tiles = sorted(tiles)
      ret += (''.join(str(x) for x in tiles) + suit.upper()) * count
    suit_to_pieces = defaultdict(list)
    for (tiles, suit), count in self.items():
      if isinstance(tiles, int):
        for _ in range(count):
          suit_to_pieces[suit].append(tiles)
    for suit in 'pmsz':
      if suit in suit_to_pieces:
        ret += ''.join(str(x) for x in sorted(suit_to_pieces[suit])) + suit
    return ret

ALL_牌: Final = Hand({(i, c): 4 for i in range(1, 10) for c in 'pms'}) + Hand({(i, 'z'): 4 for i in range(1, 8)})

def parse_hand(s: str) -> Hand:
  current = []
  hand = Hand()
  for c in s:
    if c in '123456789':
      current.append(int(c))
    elif c in 'PMSZ':
      assert len(current) == 3
      hand[(tuple(sorted(current)), c.lower())] += 1
      current.clear()
    else:
      assert c in 'pmsz'
      hand.update((x, c) for x in current)
      current.clear()
  return hand

def melt_down(hand: Hand) -> Hand:
  result = Hand()
  for (tiles, suit), count in hand.items():
    if isinstance(tiles, int):
      result[(tiles, suit)] += count
    else:
      for tile in tiles:
        result[(tile, suit)] += count
  return result

def count_pieces(hand: Hand) -> int:
  return sum(count if isinstance(tiles, int) else 3 * count for (tiles, _), count in hand.items())

def 三顺(center: int, step: int, order: str) -> Hand:
  assert len(order) == 3
  assert all(c in 'pms' for c in order)
  hand = Hand()
  def centering(i):
    return (i - 1, i, i + 1)
  for i, c in zip([-step, 0, step], order):
    hand[(centering(center + i), c)] += 1
  return hand, 1, 1

def 三刻(center: int, step: int, order: str) -> Hand:
  assert len(order) == 3
  assert all(c in 'pms' for c in order)
  hand = Hand()
  def centering(i):
    return (i, i, i)
  for i, c in zip([-step, 0, step], order):
    hand[(centering(center + i), c)] += 1
  return hand, 1, 1

def 刻子(hand: Hand) -> Hand:
  return Hand(((tile, tile, tile), color) for tile, color in hand), 4 - len(hand), 1

def 组合龙(order: str) -> Hand:
  assert len(order) == 3
  assert 'm' in order and 'p' in order and 's' in order
  return Hand(zip(range(1, 10), order * 3)), 1, 1

def 双龙会(order: str) -> Hand:
  assert len(order) == 3
  assert all(c in 'pms' for c in order)
  hand = Hand()
  hand[(5, order[0])] += 2
  for body in order[1:]:
    hand[((1, 2, 3), body)] += 1
    hand[((7, 8, 9), body)] += 1
  return hand, 0, 0

def 全不靠(order: str) -> Hand:
  assert len(order) == 3
  assert 'm' in order and 'p' in order and 's' in order
  return Hand(list(zip(range(1, 10), order * 3)) + [(i, 'z') for i in range(1, 8)]), 0, 0

def 十三幺(which: int) -> Hand:
  assert which in range(0, 13)
  pai = [(i, color) for i in (1, 9) for color in 'pms'] + [(i, 'z') for i in range(1, 8)]
  result = Hand(pai)
  result[pai[which]] += 1
  return result, 0, 0

def find_面子(hand: Hand) -> list[Hand]:
  possibles = []
  for x, c in hand.items():
    if c >= 3:
      possibles.append(Hand({x: 3}))
  for i in range(1, 8):
    for c in 'pms':
      this_hand = Hand((i + j, c) for j in range(3))
      if not (this_hand - hand):
        possibles.append(this_hand)
  if possibles:
    return possibles
  for x, c in hand.items():
    if c == 2:
      possibles.append(Hand({x: 3}))
  for color in 'pms':
    m = [i for i in range(1, 10) if hand[(i, color)] > 0]
    if len(m) > 1:
      for i, j in zip(m, m[1:]):
        if j - i == 2:
          possibles.append(Hand((i + di, color) for di in range(3)))
        elif j - i == 1:
          if i - 1 > 0:
            possibles.append(Hand((i - 1 + di, color) for di in range(3)))
          if j + 1 < 10:
            possibles.append(Hand((i + di, color) for di in range(3)))
  if possibles:
    return possibles
  for x in hand:
    possibles.append(Hand({x: 3}))
  def add_start(x, color):
    if x < 1:
      return
    if x + 2 > 9:
      return
    possibles.append(Hand((x + di, color) for di in range(3)))
  for color in 'pms':
    m = [i for i in range(1, 10) if hand[(i, color)] > 0]
    for i in m:
      for j in [i - 2, i - 1, i]:
        add_start(j, color)
  return possibles

def count_pieces_for_1面子_1雀头(hand: Hand):
  count_面子 = sum(count for (tiles, _), count in hand.items() if isinstance(tiles, tuple))
  if count_面子 > 1:
    return 5, defaultdict(set)
  if count_面子 == 1:
    other_pieces = Hand({k: v for k, v in hand.items() if not isinstance(k[0], tuple)})
    needable = defaultdict(set)
    for x, c in other_pieces.items():
      if c >= 2:
        other = other_pieces - Hand({x: 2})
        for p in other:
          needable[p] = set()
        if not other:
          needable[()] = set()
    if needable:
      return 0, needable
    for x in other_pieces:
      needable[x].update(other_pieces - Hand({x: 1}))
    return 1, needable
  pieces = 5
  needable = defaultdict(set)
  for x, c in hand.items():
    if c >= 2:
      remaining = hand - Hand({x: 2})
      for p in find_面子(remaining):
        needed = p - remaining
        useless = remaining - p
        dist = count_pieces(needed)
        if dist < pieces:
          pieces = dist
          needable = defaultdict(set)
        if dist == pieces:
          for k in useless:
            needable[k].update(needed)
          if not useless:
            needable[()].update(needed)
  for p in find_面子(hand):
    needed = p - hand
    useless = hand - p
    dist = count_pieces(needed) + 1
    if dist < pieces:
      pieces = dist
      needable = defaultdict(set)
    if dist == pieces:
      for k in useless:
        needable[k].update(needed)
        needable[k].update(k2 for k2 in useless if k2 != k)
  return pieces, needable

def dist_from_hand(hand: Hand, target: Hand, count_面子: int, count_雀头: int):
  needed_tiles = melt_down(target - hand)
  needed_in_target = needed_tiles - hand
  other_tiles = hand - target - needed_tiles
  if count_面子 == count_雀头 == 1:
    dist, rem_to_need = count_pieces_for_1面子_1雀头(other_tiles)
  else:
    assert count_面子 == count_雀头 == 0
    dist = 0
    if other_tiles:
      rem_to_need = defaultdict(set, {tile: set() for tile in other_tiles})
    else:
      rem_to_need = defaultdict(set, {(): set()})
  if count_pieces(hand) == 13:
    mixed = set()
    for need in rem_to_need.values():
      mixed.update(need)
    rem_to_need = {(): mixed}
  for need in rem_to_need.values():
    need.update(needed_in_target)
  return 3 * (4 - count_面子) + 2 * (1 - count_雀头) - (count_pieces(hand) - count_pieces(other_tiles)) + dist, rem_to_need

def all_usable_役():
  ret = []
  orders = ['smp', 'spm', 'mps', 'msp', 'psm', 'pms']
  for i in range(1, 10):
    # 三同刻
    ret.append(('三刻', i, 0, 'smp'))
  for i in range(2, 9):
    # 三色三同顺
    ret.append(('三顺', i, 0, 'smp'))
    for color in 'smp':
      # 一色三同顺
      ret.append(('三顺', i, 0, color * 3))
      # 一色三节高
      ret.append(('三刻', i, 1, color * 3))
    for order in orders:
      # 三色三节高
      ret.append(('三刻', i, 1, order))
  for i in range(3, 8):
    for order in orders:
      # 三色三步高
      ret.append(('三顺', i, 1, order))
    for color in 'smp':
      # 一色三步高
      ret.append(('三顺', i, 1, color * 3))
  for i in range(4, 7):
    for color in 'smp':
      # 一色三连环
      ret.append(('三顺', i, 2, color * 3))
  for order in orders:
    # 花龙
    ret.append(('三顺', 5, 3, order))
  for color in 'smp':
    # 清龙
    ret.append(('三顺', 5, 3, color * 3))
  for order in orders:
    # 组合龙
    ret.append(('组合龙', order))
  for color in 'smp':
    # 一色双龙会 < 必定是清一色，可能可以丢弃？
    # ret.append(('双龙会', color * 3))
    # 三色双龙会
    ret.append(('双龙会', color + 'smp'.replace(color, '')))
  for num in range(1, 5):
    # 三风刻
    ret.append(('刻子', Hand((n, 'z') for n in range(1, 5) if n != num)))
  for num in range(5, 8):
    # 双箭刻
    # ret.append(('刻子', Hand((n, 'z') for n in range(5, 8) if n != num)))
    # 无法处理需要两个面子的情况
    pass
  # 全带幺 全带五
  # 混一色 清一色 推不倒 大于五 小于五 全大 全中 全小 字一色
  for order in orders:
    ret.append(('全不靠', order))
  for i in range(13):
    ret.append(('十三幺', i))
  # 五门齐 无番和
  # 三暗刻 七对
  # 碰碰和
  return ret

ALL_USABLE_役: Final = all_usable_役()

def possibilities(hand: Hand, usable_役=ALL_USABLE_役):
  dist = [[] for _ in range(14)]
  def update(役, *args):
    func = globals()[役]
    d, rn = dist_from_hand(hand, *func(*args))
    d = min(d, 13)
    dist[d].append((rn, (役, *args)))
  for 役 in usable_役:
    update(*役)
  min_dist = min(i for i, x in enumerate(dist) if x)
  return dist[:min_dist + 2]

def which_to_remove(hand: Hand, usable_役=ALL_USABLE_役):
  dist = possibilities(hand, usable_役)
  def dist_to_rem(dist):
    rem = defaultdict(lambda: defaultdict(list))
    for rn, 役 in dist:
      for x, n in rn.items():
        for y in n:
          rem[x][y].append(役)
        if not n:
          rem[x][()].append(役)
    rem = {x: {y: z for y, z in y.items()} for x, y in rem.items()}
    return rem
  return [dist_to_rem(d) for d in dist]

def simple_count(hand: Hand):
  rem = which_to_remove(hand)
  rem = [[(x, len(y)) for x, y in x.items()] for x in rem]
  rem = [sorted(x, key=lambda x: -x[1]) for x in rem]
  return rem

def count_with_others(hand: Hand, remain: Hand):
  rem = which_to_remove(hand)
  rem = [[(x, sum(remain[yi] for yi in y)) for x, y in x.items()] for x in rem]
  rem = [sorted(x, key=lambda x: -x[1]) for x in rem]
  return rem

@dataclass
class GameTree:
  hand: Hand
  next: dict[牌, dict[牌, 'GameTree']] | list

  def choose_with_remain(self, remain: Hand):
    if isinstance(self.next, list):
      return self.next
    return {x: branch_p(branch, remain) for x, branch in self.next.items()}

  def prune(self) -> 'GameTree':
    if not isinstance(self.next, dict):
      return self
    new_next = {
      x: {y: z.prune() for y, z in ys.items()}
      for x, ys in self.next.items()
    }
    removal_candidates = [
      (x1, x2)
      for x1, ys1 in new_next.items()
      for x2, ys2 in new_next.items()
      if x1 != x2 and set(ys1).issubset(ys2)
    ]
    oneside_removal = [(x1, x2) for x1, x2 in removal_candidates if (x2, x1) not in removal_candidates]
    twoside_removal = [(x1, x2) for x1, x2 in removal_candidates if (x1, x2) not in oneside_removal and x1 < x2]
    def future_not_better_than(future1: dict[tuple[int, str], GameTree], future2: dict[tuple[int, str], GameTree]) -> bool:
      # future1 <= future2
      return all(y in future2 and not_better_than(future1[y], future2[y]) for y in future1)
    def not_better_than(tree1: GameTree, tree2: GameTree) -> bool:
      # tree1 <= tree2
      tree1_is_leaf = not isinstance(tree1.next, dict)
      tree2_is_leaf = not isinstance(tree2.next, dict)
      assert tree1_is_leaf == tree2_is_leaf
      if tree1_is_leaf:
        return True
      # forall future1 in tree1.next, exists future2 in tree2.next such that future1 <= future2
      return all(
        any(
          future_not_better_than(ys1, ys2)
          for ys2 in tree2.next.values()
        )
        for ys1 in tree1.next.values()
      )
    for x1, x2 in oneside_removal:
      if x1 not in new_next or x2 not in new_next:
        continue
      if future_not_better_than(new_next[x1], new_next[x2]):
        del new_next[x1]
    for x1, x2 in twoside_removal:
      if x1 not in new_next or x2 not in new_next:
        continue
      x1_leq_x2 = future_not_better_than(new_next[x1], new_next[x2])
      x2_leq_x1 = future_not_better_than(new_next[x2], new_next[x1])
      if x1_leq_x2 == x2_leq_x1:
        continue
      if x1_leq_x2:
        del new_next[x1]
      else:
        del new_next[x2]
    return GameTree(self.hand, new_next)

def get_futures(hand: Hand, usable_役=ALL_USABLE_役) -> GameTree:
  assert count_pieces(hand) == 14
  rem = which_to_remove(hand, usable_役)
  usable = [i for i in rem if i][0]
  return GameTree(hand, {x: {y: get_futures(Hand(hand - Hand({x: 1}) + Hand({y: 1})), 役) for y, 役 in ys.items()} for x, ys in usable.items()} if () not in usable else usable[()][()])

def hand_p(tree: GameTree, remain: Hand) -> int:
  if not isinstance(tree.next, dict):
    return 1
  return max(branch_p(branch, remain) for branch in tree.next.values())

def branch_p(branch: dict[牌, GameTree], remain: Hand) -> int:
  return sum(remain[x] * hand_p(tree, remain - Hand([x])) for x, tree in branch.items())
