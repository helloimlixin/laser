import copy
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

from src.training.rq_sample_panels import CLASSES, MEDIA_KEYS, preview_labels, save_panels


def test_requested_labels_match_actual_imagenet_class_indices():
    categories = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
    assert [categories[label].lower().replace(' ', '') for _, label in CLASSES] == [
        'cheeseburger', 'teapot', 'samoyed', 'baldeagle', 'redfox', 'hotpot', 'wombat', 'timberwolf']
    labels = preview_labels()
    for world in (1, 4, 5, 8):
        reassembled = torch.empty_like(labels)
        for rank in range(world):
            reassembled[rank::world] = labels[rank::world]
        assert torch.equal(labels, reassembled)
    assert torch.equal(labels.reshape(8, 64)[:, 0], torch.tensor([933, 849, 258, 22, 277, 926, 106, 269]))


def test_overview_first_ten_and_class_grids_are_same_unselected_images(tmp_path):
    index = torch.arange(512)
    colors = torch.stack((index % 256, index // 256, index // 64), -1).to(torch.uint8)
    images = colors[:, :, None, None].expand(-1, -1, 256, 256)
    paths = save_panels(images, tmp_path, 20)
    assert tuple(paths) == MEDIA_KEYS
    with Image.open(paths['samples/fixed']) as overview:
        assert overview.size == (2752, 2048)
        for row in range(8):
            with Image.open(paths[MEDIA_KEYS[row+1]]) as grid:
                assert grid.size == (2048, 2088)
                for image in range(64):
                    expected = tuple(colors[row*64+image].tolist())
                    assert grid.getpixel(((image%8)*256+100, 40+(image//8)*256+100)) == expected
                    if image < 10:
                        assert overview.getpixel((192+image*256+100, row*256+100)) == expected


def test_each_panel_requires_only_a_key_logged_by_this_run():
    from scripts.tools.configure_rq_sample_panels import add_sample_sections
    source = {'section':{'panelBankConfig':{'sections':[
        {'name':'FID', 'flowConfig':{}, 'panels':[{'config':{'metrics':['eval/fid']}}]}]}}}
    before = copy.deepcopy(source)
    result = add_sample_sections(source)
    assert source == before
    sections = result['section']['panelBankConfig']['sections']
    panels = [p for s in sections[1:] for p in s['panels']]
    assert [p['config']['mediaKeys'] for p in panels] == [[key] for key in MEDIA_KEYS]
    assert all('samples/continuation' not in p['config']['mediaKeys'] for p in panels)
    # Repeated configuration must not duplicate panels or the FID chart.
    assert len(add_sample_sections(result)['section']['panelBankConfig']['sections']) == 3
