import pickle

with open('image_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

# stats: list of (img_name, pos_pixels, tot_pixels)
# Sort by pos_pixels descending
stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)

# Try to find a subset that gives 10:1
# If we greedily add images with highest pos_pixels, what ratio do we get?
pos_acc = 0
tot_acc = 0
for i, (name, pos, tot) in enumerate(stats_sorted):
    pos_acc += pos
    tot_acc += tot
    ratio = (tot_acc - pos_acc) / max(1, pos_acc)
    if i % 100 == 0:
        print(f"Top {i} images: ratio = {ratio:.2f}")

print(f"Top {len(stats_sorted)} images: ratio = {(tot_acc - pos_acc) / max(1, pos_acc):.2f}")
