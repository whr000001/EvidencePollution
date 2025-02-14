import os


def main():
    os.system('CUDA_VISIBLE_DEVICES=0 python polluted_amr.py --a &')
    os.system('CUDA_VISIBLE_DEVICES=0 python polluted_amr.py --b &')
    os.system('CUDA_VISIBLE_DEVICES=1 python polluted_amr.py --c &')
    os.system('CUDA_VISIBLE_DEVICES=1 python polluted_amr.py --d &')
    os.system('CUDA_VISIBLE_DEVICES=2 python polluted_amr.py --e &')
    os.system('CUDA_VISIBLE_DEVICES=2 python polluted_amr.py --f &')
    os.system('CUDA_VISIBLE_DEVICES=3 python polluted_amr.py --g &')
    os.system('CUDA_VISIBLE_DEVICES=3 python polluted_amr.py --h &')
    os.system('CUDA_VISIBLE_DEVICES=4 python polluted_amr.py --i &')
    os.system('CUDA_VISIBLE_DEVICES=4 python polluted_amr.py --j &')


if __name__ == '__main__':
    main()
