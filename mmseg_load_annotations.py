class LoadAnnotationsDiff(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend


        self.cutouts = []
        self.cutout_class = 0
        self.cutout_dataset_path = '/home/jmr/Desktop/jmr/ucg/dataset_trains/'
        self.cutout_dataset_path2= '/home/jmr/Desktop/jmr/ucg/dataset/'
        #Load here the diffusers list
        with open(self.cutout_dataset_path+"train.txt",'r') as file:
            self.cutouts = file.readlines()

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if "filename_orig" in results:
            if results["filename_orig"] != results["filename"]:
                results['ann_info']['seg_map'].replace(results['ann_info']['seg_map'].split("/")[-1], results["filename"].split("/")[-1][:-4] +"_labelTrainIds.png" )
        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg

        """Replaces all pixels of a certain class in an RGB image with corresponding pixels from another image"""


        if np.random.randint(0,100) < 5:
            #Get cutout
            cutout_filename = np.random.choice(self.cutouts)
            cutout_filename = cutout_filename.split(' \n')[0]

            #Load Cutout
            img_bytes_cm = self.file_client.get(self.cutout_dataset_path+'ss/'+cutout_filename)
            cutout_mask = mmcv.imfrombytes(
                img_bytes_cm, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            #Load RGB
            img_bytes_crgb = self.file_client.get(self.cutout_dataset_path+'rgb/'+cutout_filename)
            cutout_rgb = mmcv.imfrombytes(
                img_bytes_crgb, flag='color', backend=self.imdecode_backend)
            #Convert mask to appropiate int
            cutout_mask = cutout_mask*16

            # Compute maximum position for random pasting
            max_y = results['img'].shape[0] - cutout_rgb.shape[0]
            max_x = results['img'].shape[1] - cutout_rgb.shape[1]


            # Randomly choose a position to paste the image
            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)

            results['img'][y:y + 512, x:x + 512][cutout_mask>0] = cutout_rgb[cutout_mask>0]
            results['gt_semantic_seg'][y:y + 512, x:x + 512][cutout_mask>0] = cutout_mask[cutout_mask>0]

        if np.random.randint(0,100) < 5:
            #Get cutout
            cutout_filename = np.random.choice(self.cutouts)
            cutout_filename = cutout_filename.split(' \n')[0]

            #Load Cutout
            img_bytes_cm = self.file_client.get(self.cutout_dataset_path2+'ss/'+cutout_filename)
            cutout_mask = mmcv.imfrombytes(
                img_bytes_cm, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
            #Load RGB
            img_bytes_crgb = self.file_client.get(self.cutout_dataset_path2+'rgb/'+cutout_filename)
            cutout_rgb = mmcv.imfrombytes(
                img_bytes_crgb, flag='color', backend=self.imdecode_backend)
            #Convert mask to appropiate int
            cutout_mask = cutout_mask*14

            # Compute maximum position for random pasting
            max_y = results['img'].shape[0] - cutout_rgb.shape[0]
            max_x = results['img'].shape[1] - cutout_rgb.shape[1]


            # Randomly choose a position to paste the image
            y = np.random.randint(0, max_y)
            x = np.random.randint(0, max_x)

            results['img'][y:y + 512, x:x + 512][cutout_mask>0] = cutout_rgb[cutout_mask>0]
            results['gt_semantic_seg'][y:y + 512, x:x + 512][cutout_mask>0] = cutout_mask[cutout_mask>0]

        results['seg_fields'].append('gt_semantic_seg')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str