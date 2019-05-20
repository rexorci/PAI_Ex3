from importlib import reload  # to get changes in code
import chriweb_a2

chriweb_a2 = reload(chriweb_a2)
ii = chriweb_a2.IntrepidIbex()

figure = 1
field = [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', 'g', '.', '.', '.', '.', '.', 'W', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 's', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'S', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'w', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', 'g', '#', '.', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', 'g', '#', '#', '.', '.', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', 'g', 'g', 'g', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']]

game_features = ii.get_features_sheep(figure, field)
#
# figure = 1
# field = [['.', '.', '.', '.', '.', 'W', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#          ['.', '#', 'r', '.', 'g', '#', '.', '.', '.', '.', '.', '.', '.', '#', 'g', '.', 'r', '#', '.'],
#          ['.', '#', '#', 'r', '#', '#', '.', '#', 'g', '.', 'g', '#', '.', '#', '#', 'r', '#', '#', '.'],
#          ['.', '.', 'r', '.', '.', '.', 'S', '.', '.', '.', '.', '.', 'g', '.', '.', '.', 'r', '.', '.'],
#          ['.', '#', 'r', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'r', '#', '.'],
#          ['.', '.', '#', '#', '#', '#', 'g', '#', 'g', '.', 'g', '#', 'g', '#', '#', '#', '#', '.', '.'],
#          ['.', '.', '#', '.', '.', '#', '.', '.', '#', '.', '#', '.', '.', '#', '.', '.', '#', '.', '.'],
#          ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#          ['.', '.', '#', '.', '.', '#', '.', '.', '#', '.', '#', '.', '.', '#', '.', '.', '#', '.', '.'],
#          ['.', '.', '#', '#', '#', '#', 'g', '#', 'g', '.', 'g', '#', 'g', '#', '#', '#', '#', '.', '.'],
#          ['.', '#', 'r', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'r', '#', '.'],
#          ['.', '.', 'r', '.', '.', '.', 'g', '.', '.', '.', '.', '.', 's', '.', '.', '.', 'r', '.', '.'],
#          ['.', '#', '#', 'r', '#', '#', '.', '#', 'g', '.', 'g', '#', '.', '#', '#', 'r', '#', '#', '.'],
#          ['.', '#', 'r', '.', 'g', '#', '.', '.', '.', '.', '.', '.', '.', '#', 'g', '.', 'r', '#', '.'],
#          ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', 'w', '.', '.', '.', '.', '.']]
# game_features = ii.get_features_wolf(figure, field)


print(game_features)
