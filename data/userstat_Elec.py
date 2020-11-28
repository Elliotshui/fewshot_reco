import pickle
from tqdm import tqdm

dat = pickle.load(open('Electronics.dat', 'rb'))

user_cnt = {}

for u in tqdm(dat['rid']):
    if u not in user_cnt:
        user_cnt[u] = 0
    user_cnt[u] += 1

users_with_cnt = {}
for u, cnt in tqdm(user_cnt.items()):
    if cnt not in users_with_cnt:
        users_with_cnt[cnt] = []
    users_with_cnt[cnt].append(u)

stat = {
    'ucnt' : user_cnt,
    'ulist': users_with_cnt
}

print(len(users_with_cnt.keys()))
pickle.dump(stat, open('Electronics.stat', 'wb'))