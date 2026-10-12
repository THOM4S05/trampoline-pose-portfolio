import argparse
import json
import sys
from pathlib import Path
from .types import TrackerConfig,AthleteSelection

def main(argv=None):
    parser=argparse.ArgumentParser(description='Standalone offline 2D trampoline joint tracker')
    commands=parser.add_subparsers(dest='command',required=True)
    track=commands.add_parser('track')
    track.add_argument('video')
    track.add_argument('--config',default='configs/default.yaml')
    track.add_argument('--device',choices=['cpu','cuda:0'])
    track.add_argument('--seed-frame',type=int,required=True)
    track.add_argument('--bbox',type=float,nargs=4,required=True,metavar=('X1','Y1','X2','Y2'))
    track.add_argument('--flight-region',type=float,nargs=4)
    track.add_argument('--json',required=True)
    track.add_argument('--csv')
    track.add_argument('--overlay')
    track.add_argument('--raw-overlay',action='store_true')
    track.add_argument('--cache')
    track.add_argument('--max-frames',type=int)
    validate=commands.add_parser('validate')
    validate.add_argument('json')
    info=commands.add_parser('inspect')
    info.add_argument('video')
    info.add_argument('--frame',type=int)
    info.add_argument('--image')
    refine=commands.add_parser('refine')
    refine.add_argument('input_json')
    refine.add_argument('--json',required=True)
    refine.add_argument('--max-gap-ms',type=float,default=100)
    refine.add_argument('--max-gap-frames',type=int,default=3)
    args=parser.parse_args(argv)
    try:
        if args.command=='validate':
            from .output import validate
            validate(json.loads(Path(args.json).read_text(encoding='utf-8')))
            print('JSON schema and frame/joint invariants pass')
        elif args.command=='inspect':
            from .video import probe,decode
            meta=probe(args.video)
            print(json.dumps(meta,indent=2))
            if args.frame is not None:
                if not args.image or args.frame<0:
                    raise ValueError('Use --frame nonnegative-index --image output.jpg')
                if Path(args.image).resolve()==Path(args.video).resolve():
                    raise ValueError('Frame image must not overwrite source video')
                import cv2
                found=False
                for timing,image in decode(args.video,meta,max_frames=args.frame+1):
                    if timing['frame_index']==args.frame:
                        Path(args.image).parent.mkdir(parents=True,exist_ok=True)
                        if not cv2.imwrite(args.image,image):
                            raise ValueError('Cannot save frame image')
                        found=True
                if not found:
                    raise ValueError('Requested frame exceeds input video')
        elif args.command=='refine':
            from .refinement import interpolate_short_gaps
            from .output import save_json,validate
            data=json.loads(Path(args.input_json).read_text(encoding='utf-8'))
            validate(data)
            data['frames']=interpolate_short_gaps(data['frames'],args.max_gap_ms,args.max_gap_frames)
            data['effective_config'].update(temporal_refinement=True,max_gap_ms=args.max_gap_ms,max_gap_frames=args.max_gap_frames)
            data['diagnostics']['refinement_validation']='experimental_unvalidated_cached_interpolation'
            data['diagnostics']['joint_inferred_fraction']=sum(j['status']=='inferred' for f in data['frames'] for j in f['joints'].values())/(len(data['frames'])*17)
            data['diagnostics']['joint_observed_fraction']=sum(j['status']=='observed' for f in data['frames'] for j in f['joints'].values())/(len(data['frames'])*17)
            from .api import TrackingResult
            TrackingResult(data).save_json(args.json)
        else:
            from dataclasses import replace
            from .api import PoseTracker
            config=TrackerConfig.from_yaml(args.config)
            if args.device:
                config=replace(config,device=args.device)
            selection=AthleteSelection(args.seed_frame,tuple(args.bbox),tuple(args.flight_region) if args.flight_region else None)
            # Validate export destinations before any expensive inference.
            destinations=[p for p in (args.json,args.csv,args.overlay,args.cache) if p]
            resolved=[Path(p).resolve() for p in destinations]
            if len(set(resolved))!=len(resolved) or Path(args.video).resolve() in resolved:
                raise ValueError('Output/cache paths must be distinct and must not overwrite input')
            def progress(stage,count,total):
                if count==1 or count%30==0 or count==total:
                    print(f'{stage}: {count}/{total or "?"}',flush=True)
            result=PoseTracker(config).process_video(args.video,selection,progress=progress,max_frames=args.max_frames,cache_path=args.cache)
            result.save_json(args.json)
            if args.csv:
                result.save_csv(args.csv)
            if args.overlay:
                result.save_video(args.overlay,args.raw_overlay)
            print(json.dumps(result.data['diagnostics'],indent=2))
        return 0
    except KeyboardInterrupt:
        print('Cancelled',file=sys.stderr)
        return 130
    except (ValueError,FileNotFoundError,RuntimeError,InterruptedError) as error:
        print(f'Error: {error}',file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())
