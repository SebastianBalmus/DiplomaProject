import { ReactNode } from 'react';
import { useMediaQuery } from '@mui/material';
import { useFormik } from 'formik';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import FormHelperText from '@mui/material/FormHelperText';
import Select from '@mui/material/Select';
import * as Yup from 'yup';
import darkTheme from '../theme'


interface TTSFormProps {
  onSubmit: (values: {text: string, model: string}) => Promise<any>
}

function TTSForm (props: TTSFormProps) {
    const isDesktop = useMediaQuery(darkTheme.breakpoints.up('md'));

    const validationSchema = Yup.object().shape({
      text: Yup.string()
      .min(10, 'Please enter at least 10 characters!')
      .required('Please enter the input text!'),
      model: Yup.string().required('Please select the model!')
    })

    const formik = useFormik({
      initialValues: {
        text: '',
        model: '',
      },
      onSubmit: props.onSubmit,
      validationSchema
    });
    return (
      <form onSubmit={formik.handleSubmit}>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100vw',
          height: '100vh',
          padding: '40px',
          gap: '20px'
        }}
      >
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            width: isDesktop ? "60%" : "100%",
            gap: '20px'
            }}
          >
            <TextField
              id="text"
              label="Text to synthesize"
              multiline
              rows={8}
              value={formik.values.text}
              onChange={formik.handleChange}
              error={formik.touched.text && Boolean(formik.errors.text)}
              helperText={formik.touched.text && formik.errors.text}
              fullWidth
            />
            <Box
              sx={{
                display: 'flex',
                width: '100%',
                flexDirection: isDesktop ? 'row' : 'column',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '20px'
              }}
            >
              <FormControl fullWidth>
              <InputLabel id="model_picker">Select your model</InputLabel>
              <Select
                labelId="model_picker"
                name="model"
                value={formik.values.model}
                label="Select your model"
                onChange={formik.handleChange}
                error={formik.touched.model && Boolean(formik.errors.model)}
              >
                  <MenuItem value={'tts_en'}>English (high-fidelity)</MenuItem>
                  <MenuItem value={'tts_ro_d_ft'}>Romanian (high-fidelity)</MenuItem>
                  <MenuItem value={'tts_ro_ft'}>Romanian - diacritics insensitive (high-fidelity)</MenuItem>
                  <MenuItem value={'tts_ro_d'}>Romanian</MenuItem>
                  <MenuItem value={'tts_ro'}>Romanian - diacritics insensitive</MenuItem>
                </Select>
                {(formik.touched.model && formik.errors.model) && (
                  <FormHelperText error>
                    {(formik.touched.model && formik.errors.model) as ReactNode}
                  </FormHelperText>
                )}
                </FormControl>
                <Button
                  fullWidth
                  size="large"
                  variant="contained"
                  type="submit"
                  sx={{
                    backgroundColor: 'primary.dark'
                  }}
                >
                Synthesize
              </Button>
            </Box>
          </Box>
      </Box>
      </form>
    )
}

export default TTSForm;
