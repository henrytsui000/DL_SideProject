from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    
    'outColor'    , # The output color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color           outColor
    Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,        0 , 'void'            , 0       , False        , True         , (111, 74,  0) , (  0,  0,  0) ),
    Label(  'ground'               ,  6 ,        0 , 'void'            , 0       , False        , True         , ( 81,  0, 81) , (  0,  0,  0) ),
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) , (244, 35,232) ),
    Label(  'sidewalk'             ,  8 ,        0 , 'flat'            , 1       , False        , False        , (244, 35,232) , (  0,  0,  0) ),
    Label(  'parking'              ,  9 ,        0 , 'flat'            , 1       , False        , True         , (250,170,160) , (  0,  0,  0) ),
    Label(  'rail track'           , 10 ,        0 , 'flat'            , 1       , False        , True         , (230,150,140) , (  0,  0,  0) ),
    Label(  'building'             , 11 ,        0 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) , (  0,  0,  0) ),
    Label(  'wall'                 , 12 ,        0 , 'construction'    , 2       , False        , False        , (102,102,156) , (  0,  0,  0) ),
    Label(  'fence'                , 13 ,        0 , 'construction'    , 2       , False        , False        , (190,153,153) , (  0,  0,  0) ),
    Label(  'guard rail'           , 14 ,        0 , 'construction'    , 2       , False        , True         , (180,165,180) , (  0,  0,  0) ),
    Label(  'bridge'               , 15 ,        0 , 'construction'    , 2       , False        , True         , (150,100,100) , (  0,  0,  0) ),
    Label(  'tunnel'               , 16 ,        0 , 'construction'    , 2       , False        , True         , (150,120, 90) , (  0,  0,  0) ),
    Label(  'pole'                 , 17 ,        0 , 'object'          , 3       , False        , False        , (153,153,153) , (  0,  0,  0) ),
    Label(  'traffic light'        , 19 ,        0 , 'object'          , 3       , False        , False        , (250,170, 30) , (  0,  0,  0) ),
    Label(  'traffic sign'         , 20 ,        0 , 'object'          , 3       , False        , False        , (220,220,  0) , (  0,  0,  0) ),
    Label(  'vegetation'           , 21 ,        0 , 'nature'          , 4       , False        , False        , (107,142, 35) , (  0,  0,  0) ),
    Label(  'terrain'              , 22 ,        0 , 'nature'          , 4       , False        , False        , (152,251,152) , (  0,  0,  0) ),
    Label(  'sky'                  , 23 ,        0 , 'sky'             , 5       , False        , False        , ( 70,130,180) , (  0,  0,  0) ),
    Label(  'person'               , 24 ,        0 , 'human'           , 6       , True         , False        , (220, 20, 60) , (  0,  0,  0) ),
    Label(  'rider'                , 25 ,        0 , 'human'           , 6       , True         , False        , (255,  0,  0) , (  0,  0,  0) ),
    Label(  'car'                  , 26 ,        0 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) , (  0,  0,  0) ),
    Label(  'truck'                , 27 ,        0 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) , (  0,  0,  0) ),
    Label(  'bus'                  , 28 ,        0 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) , (  0,  0,  0) ),
    Label(  'caravan'              , 29 ,        0 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) , (  0,  0,  0) ),
    Label(  'trailer'              , 30 ,        0 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) , (  0,  0,  0) ),
    Label(  'train'                , 31 ,        0 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) , (  0,  0,  0) ),
    Label(  'motorcycle'           , 32 ,        0 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) , (  0,  0,  0) ),
    Label(  'bicycle'              , 33 ,        0 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) , (  0,  0,  0) ),
]