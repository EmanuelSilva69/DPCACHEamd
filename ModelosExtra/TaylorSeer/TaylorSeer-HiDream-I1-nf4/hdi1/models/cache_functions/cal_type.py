# from .force_scheduler import force_scheduler

def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''

    first_step = (current['step'] < cache_dic['first_enhance'])


    fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        
    
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'taylor_cache'
        
  