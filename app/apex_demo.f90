program apex_demo
   use, intrinsic :: iso_fortran_env, only : real64, int64, output_unit, error_unit
   use apex_optimizer, only : apex_config_t, apex_state_t, apex_error_t, apex_read_config, &
                              apex_initialize, apex_optimize, apex_get_parameters, APEX_OK
   implicit none

   type(apex_config_t) :: config
   type(apex_state_t) :: state
   type(apex_error_t) :: error
   real(real64) :: initial(2), parameters(2)
   character(len=256) :: config_path

   config_path = 'apex_config.nml'
   if (command_argument_count() >= 1) call get_command_argument(1, config_path)
   initial = [2.0_real64, -1.0_real64]
   call apex_read_config(trim(config_path), config, error)
   call require_apex_ok(error, 'configuration load')
   call apex_initialize(state, initial, config, error)
   call require_apex_ok(error, 'state initialization')
   call apex_optimize(state, quadratic_gradient, 1000_int64, error)
   call require_apex_ok(error, 'quadratic optimization')
   call apex_get_parameters(state, parameters, error)
   call require_apex_ok(error, 'parameter readback')
   write(output_unit, '(a,2(1x,es16.8))') 'Apex parameters:', parameters

contains

   subroutine require_apex_ok(result, operation)
      type(apex_error_t), intent(in) :: result
      character(len=*), intent(in) :: operation
      if (result%code /= APEX_OK) then
         write(error_unit, '(a,a,a,a)') 'Apex demo ', trim(operation), ': ', trim(result%message)
         stop 1
      end if
   end subroutine require_apex_ok

   subroutine quadratic_gradient(parameters, gradient, objective, result)
      real(real64), contiguous, intent(in) :: parameters(:)
      real(real64), contiguous, intent(out) :: gradient(:)
      real(real64), intent(out) :: objective
      type(apex_error_t), intent(out) :: result
      gradient = parameters
      objective = 0.5_real64 * sum(parameters * parameters)
      result%code = APEX_OK
      result%message = ''
   end subroutine quadratic_gradient

end program apex_demo
