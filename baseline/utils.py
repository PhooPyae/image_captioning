def get_loader(
    dataset,
    batch_size = 32,
    num_workers = 4,
    shuffle = True,
    pin_memory = True
):
    pad_idx = dataset.vocab.string_to_index["<PAD>"]
    
    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn = MyCollate(pad_idx = pad_idx))
    
    return loader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def print_examples(model, device, dataset, path):
    model.eval()
    for i in range(5):
        idx = np.random.choice(dataset.df.index)
        img = dataset.df.loc[idx, 'image']
        caption = dataset.df.loc[idx, 'caption']
        image = Image.open(path+"/"+img).convert("RGB")
        transformed_image = transform(image).unsqueeze(0)
        generated_caption = " ".join(model.caption_image(transformed_image.to(device), dataset.vocab))
        print(f'Ground Truth Caption: {caption}')
        print(f'Generated Caption: {generated_caption}')