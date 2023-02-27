import logging
import platform
import os

import azure.functions as func
import di_log
from di_log import p

import di_detect_from_image

import cv2

try:
    p('OS: {} {}'.format(platform.system(), os.name))
    p('--------------------')
    p('========= di_detect_from_image =========')
    args_model = 'saved_model' 
    fn = 'test.jpg'
    p('detection -1')
    p('File {} exists? : {}'.format(fn, os.path.exists(fn)))
    timg = cv2.imread(fn)
    p('timg.shape: {}'.format(timg.shape))

    p('detection 0')
    detection_model = di_detect_from_image.load_model(args_model)
    p('detection A')
    df, img_res = di_detect_from_image.run_inference_for_single_file(detection_model, fn)
    p('detection B')
    p('len(df): {}'.format(len(df)))
    p('detection END')
    # print(df)
    df.to_csv('res.txt', sep='\t', index=False)
    cv2.imwrite('res_img.jpg', img_res)



except Exception as ex:
    # evtl. alte Messung ohne notwendigen DB-Eintrag, z.B. fehlendes summary_results in DB
    p('DI Error')
    p('    ErrorNo: 220')
    p('    General exception (Exception): {}'.format(ex))

except KeyboardInterrupt:
    p('DI Error')
    p('    ErrorNo: 224')
    p('    Exception KeyboardInterrupt.')

except SystemExit:
    p('DI Error')
    p('    ErrorNo: 73')
    p('    Exception SystemExit.')

except:
    p('DI Error')
    p('    ErrorNo: 79')
    p('    Exception except.')

print(di_log.dilog.message_dict)
