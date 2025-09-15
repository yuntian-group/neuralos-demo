 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 1 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 2 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 4 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 8 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 16 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 32 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 64 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions_teacher_forcing.py  --prefix 2_074k --teacher-forcing --steps 128 --split train



 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 1 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 2 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 4 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 8 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 16 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 32 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 64 --split train
 CUDA_VISIBLE_DEVICES=1  python render_sessions.py  --prefix 2_074k --steps 128 --split train
