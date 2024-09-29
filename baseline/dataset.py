class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform = None, freq_threshold = 5):
        self.root_dir = root_dir
        df = pd.read_csv(captions_file)
        
        df['image'] = df['image_name']
        df['caption'] = df['comment']
        self.df = df.loc[:,['image', 'caption']]
        
        self.transform = transform
    
        self.images = self.df['image']
        self.captions = self.df['caption']
        
        #init vocab and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
#         print(caption)
        img_id = self.images[idx]
        image = Image.open('/kaggle/input/flickr30k/flickr30k_images/'+img_id).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        tokenized_caption = [self.vocab.string_to_index["<SOS>"]]
        tokenized_caption += self.vocab.tokenize(caption)
        tokenized_caption.append(self.vocab.string_to_index["<EOS>"])
#         print(tokenized_caption)
#         print('--------------')
#         print()
        
        return image, torch.tensor(tokenized_caption)
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)

        return images, targets
    