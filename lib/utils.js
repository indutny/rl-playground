export function shuffle(list) {
  for (let i = list.length - 1; i >= 0; i--) {
    const j = Math.round(Math.random() * i);
    const t = list[i];
    list[i] = list[j];
    list[j] = t;
  }
}
