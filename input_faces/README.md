# Reference faces

Drop the people you want recognised in here. **One photo per person is enough.**

Two layouts, and you can mix them freely:

```
input_faces/
├── Mohammed.jpg          # one photo  ->  person "Mohammed"
├── Sara.png              # one photo  ->  person "Sara"
└── Ahmed/                # a folder   ->  person "Ahmed"
    ├── front.jpg
    └── side.jpg
```

The file name (or folder name) is the label drawn on screen, so name the files
after the people.

## What makes a good reference photo

- One clear, front-facing face, reasonably sharp and at least ~112 px wide.
- Normal lighting, no heavy filters, eyes visible (sunglasses hurt a lot).
- If a photo contains several people, the **largest** face is the one enrolled -
  so crop group photos before using them.

Extra photos are optional but always help: different angles, lighting and years
make recognition noticeably more forgiving.

The embeddings are cached in `.input_faces.gallery.npz` next to this folder and
rebuilt automatically whenever you add, remove or change a photo.
