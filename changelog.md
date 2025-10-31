# 352023

Error message calling 
  model = KerasRegressor(build_fn=design_model(X_train))
  grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)


```
Traceback (most recent call last):
  File "./main_mlsubop.py", line 35, in <module>
    main()
  File "./main_mlsubop.py", line 31, in main
    do_randomized_search(X_train, y_train)
  File "C:\Users\Guanchu He\Desktop\mlsuboptimal\modeltune.py", line 14, in do_randomized_search
    grid_result = grid.fit(X_train, y_train, verbose = 0)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\sklearn\model_selection\_search.py", line 788, in fit
    base_estimator = clone(self.estimator)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\sklearn\base.py", line 89, in clone
    new_object_params[name] = clone(param, safe=False)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\sklearn\base.py", line 70, in clone
    return copy.deepcopy(estimator)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\copy.py", line 153, in deepcopy
    y = copier(memo)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\keras\engine\training.py", line 377, in __deepcopy__
    new = pickle_utils.deserialize_model_from_bytecode(
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\keras\saving\pickle_utils.py", line 47, in deserialize_model_from_bytecode
    model = save_module.load_model(temp_dir)
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\keras\utils\traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "C:\pogramming\anaconda3\envs\mytensor\lib\site-packages\tensorflow\python\saved_model\load.py", line 933, in load_partial
    raise FileNotFoundError(
FileNotFoundError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ram://4926599b-7007-495e-90ba-a97d91b8668c/variables/variables
 You may be trying to load on a different device from the computational device. Consider setting the `experimental_io_device` option in `tf.saved_model.LoadOptions` to the io_device such as '/job:localhost'.
 
 ```