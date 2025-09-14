import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Import the OVAM library
from ovam import StableDiffusionHooker
from ovam.utils import set_seed, get_device
from ovam.optimize import optimize_embedding
from ovam.utils.dcrf import densecrf

from tqdm import tqdm
model_id = "runwayml/stable-diffusion-v1-5"
device = get_device()

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def generateprompt():

    train_types = [
    "steam locomotive",
    "diesel locomotive",
    "electric locomotive",
    "high-speed train",
    "bullet train",
    "freight train",
    "passenger train",
    "subway train",
    "light rail train",
    "monorail",
    "tram",
    "commuter train",
    "double-decker train",
    "maglev train",
    "vintage train",
    "intercity train",
    "sleeper train",
    "cargo train",
    "railcar",
    "metro train"
    ]
    
    locations = [
        "at the train station",
        "on the tracks",
        "in a tunnel",
        "on a bridge",
        "in the countryside",
        "in a rural area",
        "in the city center",
        "at a crossing",
        "in the mountains",
        "in the desert",
        "along the coast",
        "on an elevated track",
        "in an industrial area",
        "at a railway yard",
        "by a river",
        "in a snowy landscape",
        "in an urban area",
        "at a platform",
        "in a forested area",
        "in a valley"
    ]
    class_type = np.random.choice(train_types,1)[0]
    
    location = np.random.choice(locations,1)[0]
    style = ',cityscapes, ego camera, color'
    return class_type + " " + location + style, 1 if len(class_type.split(" ")) == 1 else 2

for step in tqdm(range(4000)): #range(10000):
    prompt, token_position = generateprompt()

    #print("Generating image for prompt: \"{}\" ".format(prompt))
    #print("Token position: {}".format(token_position))

    #only bus prompt:
    #print(prompt.split("bus"))
    prompt_eval = "train " + prompt.split("train ")[-1]
    #print("Evaluation prompt:{}".format(prompt_eval))

    with StableDiffusionHooker(pipe) as hooker:
        set_seed(step)
        out = pipe(prompt=prompt,negative_prompt='grayscale, artistic, painting')
        image = out.images[0]


    ovam_evaluator = hooker.get_ovam_callable(
        expand_size=(512, 512)
    )  # Here you can configure the OVAM evaluator (aggregation, activations, size, ...)

    

    with torch.no_grad():
        attention_maps = ovam_evaluator(prompt_eval)
        attention_maps = attention_maps[0].cpu().numpy() # (8, 512, 512)


    import torch
    from transformers import SamModel, SamProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    #Input the image to sam
    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    attention_map = attention_maps[1]
    #Get sam mask
    filtered_mask = densecrf(np.array(image), (attention_map / attention_map.max()) > 0.6)
    idx = np.argwhere(filtered_mask == 1)

    #Lets get the center

    x_max = np.amax(idx[:,0])
    x_min = np.amin(idx[:,0])

    y_max = np.amax(idx[:,1])
    y_min = np.amin(idx[:,1])
    point_x = x_min + (x_max-x_min)/2
    point_y =y_min + (y_max-y_min)/2
    input_points = [[point_y,point_x]]
    #generate additional points for extra acc
    try:
        for i in range(2):
            input_points.append([point_y,point_x + pow(-1,i) * 25])
            input_points.append([point_y + pow(-1,i) *50,point_x ])

    except:
        continue
    max_energy = np.argwhere(attention_map*filtered_mask ==np.max(attention_map))
    if len(max_energy)>1:
        input_points.append([max_energy[0][1],max_energy[0][0]])
    

    inputs = processor(image, input_points=[input_points], return_tensors="pt").to(device)
    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores


    highest_score_mask = masks[0].cpu().detach().numpy()[0][np.argmax(scores.cpu().numpy())]


    dataset_path = 'dataset_trains'
    #Save image, mask, prompt and seed
    image.save('{}/rgb/{}.png'.format(dataset_path,step))
    Image.fromarray(highest_score_mask).save('{}/ss/{}.png'.format(dataset_path,step))

    with open('{}/prompt_lists.txt'.format(dataset_path), 'a') as promptlist:
        promptlist.write('{} \n'.format(prompt))

    #Get suggested points from mask
    # fig, axes = plt.subplots(1,4,figsize=(20,5))

    # axes[0].set_title("Synthetized image")
    # axes[0].imshow(image)
    # axes[1].set_title("Bus Mask")
    # axes[1].imshow(attention_map)
    # axes[2].set_title("CRF Attention Map")
    # for point in input_points:
    #     axes[2].scatter(point[0],point[1],marker='*')
    # axes[2].imshow(filtered_mask)
    # axes[3].set_title("Sam mask score: {}".format(np.max(scores.cpu().numpy())))
    # axes[3].imshow(highest_score_mask)
    # plt.show()
    # plt.close()
