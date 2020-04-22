import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval

from PIL import Image
import matplotlib.pyplot as plt
import pickle

# Number of pics to display
nb_pics = 5

# Parameters
data_folder = 'final_dataset_flickr'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = './BEST_bleu4checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

word_map_file = 'WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file,map_location = device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

nlgeval = NLGEval(metrics_to_omit=["METEOR"])  # loads the evaluator

# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
#vocab_size = len(word_map)
vocab_size=9490


test_ids_path = os.path.join(data_folder, 'TEST_GENOME_DETS_' + data_name + '.json')
with open(test_ids_path, 'r') as j:
    test_ids = json.load(j)

ids=[]

def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST'),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    
    nb = 0
    
    for i, (image_features, caps, caplens, allcaps,objdet) in enumerate(loader):

        k = beam_size

        # Move to GPU device, if available
        image_features = image_features.to(device)  # (1, 3, 256, 256)
        image_features_mean = image_features.mean(1)
        image_features_mean = image_features_mean.expand(k,2048)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)
        
        h1 = h1.to(device)
        h2 = h2.to(device)
        c1 = c1.to(device)
        c2 = c2.to(device)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            
            h1,c1 = decoder.top_down_attention(
                torch.cat([h2,image_features_mean,embeddings], dim=1),
                (h1,c1))  # (batch_size_t, decoder_dim)
            attention_weighted_encoding = decoder.attention(image_features,h1)
            h2,c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding,h1], dim=1),(h2,c2))

            scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 200:
                break
            step += 1
        
        
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        
        # References
        img_caps = allcaps[0].tolist()

        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        #print(img_caps)
        references.append(img_caps)

        # Hypotheses
        hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypothesis = ' '.join(hypothesis)
        #print(hypothesis)
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)
        
        ids.append(objdet[1])
        
        nb += 1
        
        if nb>10*nb_pics:
            break
    
    
    # Display the pictures and captions
    karpathy_json_path='data/caption_datasets/dataset_flickr30k.json'
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
        
    with open(os.path.join('final_dataset_flickr','val_flickr_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)
    
    
    for k in range(0,len(ids),5):
        for key,val in val_data.items():
            if val == int(ids[k]):
                ind2=key

        image_name=data['images'][ids[k]]['filename']

        for img in data['images']:
            if img['imgid']==ind2:
                image_name=img['filename']
                
        path1='./data/flickr30k_train/'+image_name
        path2='./data/flickr30k_test/'+image_name
        
        try:        
            im = Image.open(path1)
            plt.imshow(im)
            plt.show()
        except:
            im = Image.open(path2)
            plt.imshow(im)
            plt.show()

        print("Predicted caption:",hypotheses[k])
        print("Reference captions: -",references[k][0])
        for caption_i  in range(1,5):
            print("                    -",references[k][5*i])
    
    
    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)
    return metrics_dict


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    beam_size = 5
    metrics_dict = evaluate(beam_size)
    print(metrics_dict)
